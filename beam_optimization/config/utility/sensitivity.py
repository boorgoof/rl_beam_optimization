"""
Sensitivity analysis for ADIGE beam line parameters.

Computes one fixed sensitivity value per parameter from TraceWin finite
differences. For each parameter, starting from a small family-appropriate
delta, it runs `repeats` common-random-number (CRN) pairs — same seed for
p+delta and p-delta within a pair, different seed across pairs — and takes
the median of the paired differences:

    d_i = score(p+delta) - score(p-delta)     [same seed within a pair]
    sensitivity_p = TARGET_SCORE_CHANGE * 2*delta / |median(d_i)|

The median of the differences is inverted once, rather than taking the
median of per-pair sensitivities (which blows up whenever a single d_i is
near zero). If the differences are too dispersed across seeds (relative
spread = stdev(d_i)/|median(d_i)| above --spread-threshold) or too close to
zero, delta is escalated (`delta *= escalation_factor`) and remeasured, up to
--max-iterations or the parameter's hardware bounds. Among the deltas tried,
the smallest one that reaches a stable measurement is kept — not necessarily
the largest.

The script always returns a numeric sensitivity for every parameter, along
with a status describing how it was obtained ("stable", "unstable",
"single_pair", "weak_signal", "hardware_limited", or "fallback_old_value" if
every simulation failed). With TARGET_SCORE_CHANGE = 1, sensitivity is the
parameter change that moves the score by about 1 point.

Run as a script:
    python -m beam_optimization.config.utility.sensitivity

Prints one table and does not modify any files.
"""
from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from beam_optimization.config.adige import (
    BEAM_STATE_FEATURES,
    PARAMETERS,
    default_params,
)


TARGET_SCORE_CHANGE: float = 1.0
CHECKPOINT_VERSION: int = 5
NPART_RATIO_INDEX: int = BEAM_STATE_FEATURES.index("npart_ratio")
STATUS_RANK = {"stable": 0, "unstable": 1, "single_pair": 2, "weak_signal": 3}


def _set_seed(simulator, seed: int) -> None:
    """Fix the TraceWin random seed when the simulator supports it."""
    if hasattr(simulator, "tracewin_params"):
        simulator.tracewin_params["random_seed"] = int(seed)


def _initial_npart_ratio(result) -> Optional[float]:
    if not result.success or result.beam_states is None:
        return None
    if len(result.beam_states) == 0:
        return None
    return float(result.beam_states[0, NPART_RATIO_INDEX])


def _verify_tracewin_particles(result, expected_npart: int, particle_key: str) -> None:
    """Fail early if TraceWin ignored the requested particle count."""
    ratio = _initial_npart_ratio(result)
    if ratio is None:
        return
    if 0.95 <= ratio <= 1.05:
        return

    apparent_npart = ratio * expected_npart
    raise RuntimeError(
        "TraceWin particle-count override appears to be ignored or inconsistent. "
        f"Requested {particle_key}={expected_npart}, but the first output stage has "
        f"npart_ratio={ratio:.4f} (about {apparent_npart:.0f} particles when "
        f"normalized by {expected_npart}). Try --tracewin-particle-key Nbr_part "
        "or inspect the TraceWin command-line option name."
    )


def probe_noise(
    simulator,
    *,
    base_seed: int = 42,
    n_probes: int = 5,
    expected_initial_npart: Optional[int] = None,
    tracewin_particle_key: str = "nbr_part1",
    verify_particles: bool = True,
    verbose: bool = True,
) -> Optional[float]:
    """Measure nominal-score noise and fixed-seed reproducibility.

    Purely informational (printed for context) — does not gate sensitivity
    acceptance, which instead relies on the dispersion of the paired
    differences themselves.
    """
    if n_probes <= 0:
        return None
    defaults = default_params()
    particles_verified = False

    if verbose:
        print(f"Noise probe: determinism check (2 runs, seed={base_seed}) ...", flush=True)
    same_seed_scores: List[float] = []
    for _ in range(2):
        _set_seed(simulator, base_seed)
        res = simulator.simulate(copy.copy(defaults))
        if res.success:
            if (
                verify_particles
                and not particles_verified
                and expected_initial_npart is not None
            ):
                _verify_tracewin_particles(res, expected_initial_npart, tracewin_particle_key)
                particles_verified = True
            same_seed_scores.append(float(res.score_val))

    if len(same_seed_scores) == 2:
        drift = abs(same_seed_scores[0] - same_seed_scores[1])
        if drift == 0.0:
            print("  TraceWin is deterministic at fixed seed (identical scores)")
        elif drift < 0.05:
            print(
                f"  TraceWin reproducible at fixed seed up to thread jitter "
                f"(|delta score| = {drift:.2e})"
            )
        else:
            print(
                f"  WARNING: TraceWin is NOT reproducible at fixed seed "
                f"(|delta score| = {drift:.4f} between identical runs).\n"
                f"  CRN cancellation will be partial; rerun with --tracewin-threads 1."
            )
    else:
        print("  WARNING: determinism probe failed (TraceWin run did not succeed).")

    if verbose:
        print(f"Noise probe: sigma_noise over {n_probes} seeds ...", flush=True)
    scores: List[float] = []
    for i in range(n_probes):
        _set_seed(simulator, base_seed + 1000 + i)
        res = simulator.simulate(copy.copy(defaults))
        if res.success:
            if (
                verify_particles
                and not particles_verified
                and expected_initial_npart is not None
            ):
                _verify_tracewin_particles(res, expected_initial_npart, tracewin_particle_key)
                particles_verified = True
            scores.append(float(res.score_val))

    if len(scores) < 2:
        print("  WARNING: noise probe failed; sigma_noise unknown.")
        return None
    sigma = statistics.pstdev(scores)
    print(
        f"  nominal score = {statistics.fmean(scores):.4f} +/- {sigma:.4f} "
        f"(n={len(scores)} seeds)"
    )
    return sigma


def compute_sensitivity(
    simulator,
    *,
    repeats: int = 10,
    escalation_factor: float = 3.0,
    max_iterations: int = 8,
    min_score_diff: float = 0.5,
    spread_threshold: float = 0.5,
    base_seed: int = 42,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Compute one fixed sensitivity value per parameter.

    For each parameter, escalates delta (starting small, family-appropriate)
    until the median of `repeats` CRN paired differences is both away from
    zero and stable across seeds (relative spread <= spread_threshold).
    Always returns a numeric sensitivity per parameter — falling back to the
    value already in adige.py if no pair ever succeeds.
    """
    if escalation_factor <= 1.0:
        raise ValueError(f"escalation_factor must be > 1.0, got {escalation_factor}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if min_score_diff <= 0:
        raise ValueError(f"min_score_diff must be > 0, got {min_score_diff}")
    if spread_threshold <= 0:
        raise ValueError(f"spread_threshold must be > 0, got {spread_threshold}")

    # Settings that must match exactly to resume a checkpoint.
    config = {
        "escalation_factor": float(escalation_factor),
        "max_iterations": int(max_iterations),
        "repeats": int(repeats),
        "min_score_diff": float(min_score_diff),
        "spread_threshold": float(spread_threshold),
        "base_seed": int(base_seed),
    }

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    records: Dict[str, dict] = {}
    if checkpoint is not None and resume:
        records = _load_checkpoint(checkpoint, config)

    defaults = default_params()
    n_total = _estimate_total_runs(max_iterations, repeats)
    run_count = 0

    def _checkpoint_save() -> None:
        if checkpoint is not None:
            _save_checkpoint(checkpoint, config, records)

    for p in PARAMETERS:
        if p.name in records:
            if verbose:
                print(
                    f"  [resume] {p.name:<14} already present "
                    f"(status={records[p.name].get('status')})",
                    flush=True,
                )
            continue

        run_count, record = _search_sensitivity_for_param(
            simulator,
            p,
            defaults,
            repeats=repeats,
            escalation_factor=escalation_factor,
            max_iterations=max_iterations,
            min_score_diff=min_score_diff,
            spread_threshold=spread_threshold,
            base_seed=base_seed,
            run_count=run_count,
            n_total=n_total,
            verbose=verbose,
        )
        records[p.name] = record
        _checkpoint_save()

    return records


def _search_sensitivity_for_param(
    simulator,
    p,
    defaults: Dict[str, float],
    *,
    repeats: int,
    escalation_factor: float,
    max_iterations: int,
    min_score_diff: float,
    spread_threshold: float,
    base_seed: int,
    run_count: int,
    n_total: int,
    verbose: bool,
) -> tuple[int, dict]:
    delta = _initial_delta(p)
    candidates: List[dict] = []
    hw_limited = False

    if verbose:
        print(f"\n{p.name}: starting delta = {_format_delta(delta)}", flush=True)

    for _iteration in range(1, max_iterations + 1):
        if not _is_valid_central_delta(p, delta):
            hw_limited = True
            if verbose:
                print(
                    f"    delta={_format_delta(delta)} would exceed hardware bounds; stopping"
                )
            break

        run_count, diffs = _measure_at_delta(
            simulator,
            p,
            defaults,
            delta,
            repeats=repeats,
            base_seed=base_seed,
            run_count=run_count,
            n_total=n_total,
            verbose=verbose,
        )
        if diffs is None:
            if verbose:
                print(f"    delta={_format_delta(delta)}: all {repeats} repeats failed on TraceWin")
            delta *= escalation_factor
            continue

        n = len(diffs)
        d_med = statistics.median(diffs)

        if abs(d_med) < min_score_diff:
            relative_spread = statistics.stdev(diffs) / abs(d_med) if n >= 2 and d_med != 0 else None
            candidates.append(
                {
                    "delta": delta,
                    "median_diff": d_med,
                    "n_pairs": n,
                    "relative_spread": relative_spread,
                    "status": "weak_signal",
                }
            )
            if verbose:
                print(
                    f"    delta={_format_delta(delta)}: median_diff={d_med:.4f} "
                    "too close to zero; escalating"
                )
            delta *= escalation_factor
            continue

        if n == 1:
            candidates.append(
                {
                    "delta": delta,
                    "median_diff": d_med,
                    "n_pairs": n,
                    "relative_spread": None,
                    "status": "single_pair",
                }
            )
            if verbose:
                print(f"    delta={_format_delta(delta)}: only 1 valid pair; escalating")
            delta *= escalation_factor
            continue

        relative_spread = statistics.stdev(diffs) / abs(d_med)
        status = "stable" if relative_spread <= spread_threshold else "unstable"
        candidates.append(
            {
                "delta": delta,
                "median_diff": d_med,
                "n_pairs": n,
                "relative_spread": relative_spread,
                "status": status,
            }
        )
        if verbose:
            print(
                f"    delta={_format_delta(delta)}: median_diff={d_med:.4f} n={n} "
                f"relative_spread={relative_spread:.2f} -> {status}"
            )
        if status == "stable":
            break
        delta *= escalation_factor

    if not candidates:
        if verbose:
            print(f"    WARNING: no valid measurement for {p.name}; keeping old sensitivity")
        return run_count, {
            "sensitivity": p.sensitivity,
            "delta": None,
            "median_diff": None,
            "n_pairs": 0,
            "relative_spread": None,
            "status": "fallback_old_value",
        }

    best = min(
        candidates,
        key=lambda c: (
            STATUS_RANK[c["status"]],
            c["relative_spread"] if c["relative_spread"] is not None else float("inf"),
        ),
    )
    if hw_limited and best["status"] != "stable":
        best["status"] = "hardware_limited"

    effective_diff = max(abs(best["median_diff"]), min_score_diff)
    best["sensitivity"] = TARGET_SCORE_CHANGE * 2.0 * abs(best["delta"]) / effective_diff

    if verbose:
        print(
            f"    SELECTED delta={_format_delta(best['delta'])} "
            f"sensitivity={best['sensitivity']:.6e} status={best['status']}"
        )

    return run_count, best


def _measure_at_delta(
    simulator,
    p,
    defaults: Dict[str, float],
    delta: float,
    *,
    repeats: int,
    base_seed: int,
    run_count: int,
    n_total: int,
    verbose: bool,
) -> tuple[int, Optional[List[float]]]:
    """Run up to `repeats` CRN pairs at a fixed delta, tolerating partial failures.

    Returns the list of valid paired differences (possibly shorter than
    `repeats`), or None if every pair failed.
    """
    diffs: List[float] = []
    for repeat_idx in range(repeats):
        seed = base_seed + repeat_idx
        run_count, diff = _run_pair(
            simulator,
            p,
            defaults,
            delta,
            seed=seed,
            run_count=run_count,
            n_total=n_total,
            label="measure",
            verbose=verbose,
        )
        if diff is not None:
            diffs.append(diff)
    return run_count, (diffs if diffs else None)


def _run_pair(
    simulator,
    p,
    defaults: Dict[str, float],
    delta: float,
    *,
    seed: int,
    run_count: int,
    n_total: int,
    label: str,
    verbose: bool,
) -> tuple[int, Optional[float]]:
    params_plus = copy.copy(defaults)
    params_minus = copy.copy(defaults)
    params_plus[p.key] = p.default + delta
    params_minus[p.key] = p.default - delta

    _set_seed(simulator, seed)
    run_count += 1
    if verbose:
        print(
            f"  [{run_count:4d}/{n_total}] {p.name:<14} +delta "
            f"({label}, delta={_format_delta(delta)}, seed={seed})",
            flush=True,
        )
    res_plus = simulator.simulate(params_plus)

    _set_seed(simulator, seed)
    run_count += 1
    if verbose:
        print(
            f"  [{run_count:4d}/{n_total}] {p.name:<14} -delta "
            f"({label}, delta={_format_delta(delta)}, seed={seed})",
            flush=True,
        )
    res_minus = simulator.simulate(params_minus)

    if res_plus.success and res_minus.success:
        return run_count, float(res_plus.score_val) - float(res_minus.score_val)
    return run_count, None


def _initial_delta(p) -> float:
    """Smallest physically-motivated starting delta for this parameter's family."""
    if p.name.startswith("AD.SO"):
        return 1e-4
    if p.name.startswith("AD.MS"):
        return 1e-6
    if p.name.startswith("AD.D"):
        return 1e-7
    if p.name.startswith("AD.1EQ") or p.name.startswith("AD.EM"):
        return 0.1
    return max(abs(float(p.default)), 1.0) * 1e-5


def _is_valid_central_delta(p, delta: float) -> bool:
    lower = p.hw_min
    upper = p.hw_max
    if p.name.startswith("AD.SO") and lower is None:
        lower = 0.0
    minus = p.default - delta
    plus = p.default + delta
    if lower is not None and minus < lower:
        return False
    if upper is not None and plus > upper:
        return False
    return True


def _estimate_total_runs(max_iterations: int, repeats: int) -> int:
    """Upper bound on TraceWin runs, assuming every parameter uses all iterations."""
    return len(PARAMETERS) * max_iterations * repeats * 2


def _format_delta(value: float) -> str:
    return f"{value:.3g}"


def _load_checkpoint(checkpoint_path: Path, config: dict) -> Dict[str, dict]:
    if not checkpoint_path.exists():
        return {}
    with open(checkpoint_path) as f:
        payload = json.load(f)

    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has version {payload.get('version')}, "
            f"this script writes version {CHECKPOINT_VERSION}. Use --no-resume "
            "or a new --checkpoint."
        )
    saved_config = payload.get("config", {})
    if saved_config != config:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was created with config={saved_config}, "
            f"but this run uses config={config}. Use --no-resume or a new --checkpoint."
        )

    return payload.get("records", {})


def _save_checkpoint(checkpoint_path: Path, config: dict, records: Dict[str, dict]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "config": config,
        "records": records,
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(checkpoint_path)


def print_sensitivity_report(records: Dict[str, dict]) -> None:
    """Print a single table with one fixed sensitivity value per parameter."""
    header = (
        f"{'Parameter':<14} {'default':>12} {'old_sens':>12} {'new_sens':>12} "
        f"{'delta':>12} {'median_diff':>12} {'pairs':>6} {'spread':>8} {'status':>18}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for p in PARAMETERS:
        record = records.get(p.name, {})
        new_sens = record.get("sensitivity")
        delta = record.get("delta")
        median_diff = record.get("median_diff")
        n_pairs = record.get("n_pairs")
        relative_spread = record.get("relative_spread")
        status = record.get("status", "n/a")

        new_sens_str = "-" if new_sens is None else f"{new_sens:.4e}"
        delta_str = "-" if delta is None else _format_delta(delta)
        median_diff_str = "-" if median_diff is None else f"{median_diff:.4f}"
        pairs_str = "-" if n_pairs is None else str(n_pairs)
        spread_str = "-" if relative_spread is None else f"{relative_spread * 100:.0f}%"

        row = (
            f"{p.name:<14} {p.default:>12.6g} {p.sensitivity:>12.4e} {new_sens_str:>12} "
            f"{delta_str:>12} {median_diff_str:>12} {pairs_str:>6} {spread_str:>8} "
            f"{status:>18}"
        )
        print(row)

    print(sep)
    print(
        "status: stable = converged with low dispersion; unstable = escalated to "
        "max_iterations still above spread_threshold; single_pair = only one CRN pair "
        "ever succeeded; weak_signal = median difference too close to zero; "
        "fallback_old_value = every pair failed, kept the adige.py value; "
        "hardware_limited = hit the hardware bound before stabilizing."
    )


if __name__ == "__main__":
    import argparse

    from beam_optimization.config.paths import (
        DEFAULT_SENSITIVITY_CALC_DIR,
        DEFAULT_SENSITIVITY_CHECKPOINT,
        DEFAULT_TRACEWIN_INI,
    )
    from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import (
        TraceWinSimulator,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Compute ADIGE parameter sensitivity from TraceWin finite differences. "
            f"Sensitivity means parameter change for {TARGET_SCORE_CHANGE:g} score point. "
            "Always prints one fixed sensitivity value per parameter, with a status "
            "describing how reliable it is. Does NOT modify adige.py."
        )
    )
    parser.add_argument("--ini", default=str(DEFAULT_TRACEWIN_INI))
    parser.add_argument("--calc-dir", default=str(DEFAULT_SENSITIVITY_CALC_DIR))
    parser.add_argument(
        "--escalation-factor", type=float, default=3.0,
        help="Multiply delta by this factor when the measurement is not yet stable (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=8,
        help="Max escalation steps per parameter before giving up (default: %(default)s)",
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
        help="CRN pairs evaluated per delta attempt (default: %(default)s)",
    )
    parser.add_argument(
        "--min-score-diff", type=float, default=0.5,
        help="Median |delta score| below this is treated as a weak signal (default: %(default)s)",
    )
    parser.add_argument(
        "--spread-threshold", type=float, default=0.5,
        help=(
            "Max relative dispersion (stdev/|median|) of the paired differences "
            "accepted as stable (default: %(default)s)"
        ),
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--noise-probes", type=int, default=5,
        help="Informational only: nominal-score noise probes, do not gate results (default: %(default)s)",
    )
    parser.add_argument("--tracewin-threads", type=int, default=None, metavar="N")
    parser.add_argument(
        "--tracewin-particles",
        type=int,
        default=10000,
        help="TraceWin macro-particles for sensitivity only (default: %(default)s)",
    )
    parser.add_argument(
        "--tracewin-particle-key",
        default="nbr_part1",
        help="TraceWin CLI key used to override particle count (default: %(default)s)",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--checkpoint", default=str(DEFAULT_SENSITIVITY_CHECKPOINT))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    tracewin_params = {args.tracewin_particle_key: int(args.tracewin_particles)}

    total_runs = _estimate_total_runs(args.max_iterations, args.repeats)
    probe_runs = 2 + max(0, args.noise_probes) if args.noise_probes > 0 else 0

    print(f"TraceWin project      : {args.ini}")
    print(f"Calc dir              : {args.calc_dir}")
    print(f"Checkpoint            : {args.checkpoint}")
    print(f"Resume                : {not args.no_resume}")
    print(f"Escalation factor     : {args.escalation_factor}")
    print(f"Max iterations        : {args.max_iterations}")
    print(f"Repeats (CRN)         : {args.repeats}")
    print(f"Min score diff        : {args.min_score_diff}")
    print(f"Spread threshold      : {args.spread_threshold}")
    print(f"Base seed             : {args.base_seed}")
    print(f"TraceWin threads      : {args.tracewin_threads or 'all CPUs'}")
    print(f"TraceWin particles    : {args.tracewin_particles}")
    print(f"TraceWin particle key : {args.tracewin_particle_key}")
    print(f"Target score change   : {TARGET_SCORE_CHANGE:g}")
    print(f"Parameters            : {len(PARAMETERS)}")
    print(f"Total TW runs         : up to {total_runs} (worst case) + {probe_runs} probe runs")
    print()

    simulator = TraceWinSimulator(
        project_file=args.ini,
        calc_dir=args.calc_dir,
        timeout=args.timeout,
        tracewin_params=tracewin_params,
        num_threads=args.tracewin_threads,
        initial_npart=args.tracewin_particles,
    )

    probe_noise(
        simulator,
        base_seed=args.base_seed,
        n_probes=args.noise_probes,
        expected_initial_npart=args.tracewin_particles,
        tracewin_particle_key=args.tracewin_particle_key,
    )

    records = compute_sensitivity(
        simulator,
        repeats=args.repeats,
        escalation_factor=args.escalation_factor,
        max_iterations=args.max_iterations,
        min_score_diff=args.min_score_diff,
        spread_threshold=args.spread_threshold,
        base_seed=args.base_seed,
        checkpoint_path=args.checkpoint,
        resume=not args.no_resume,
        verbose=True,
    )

    print_sensitivity_report(records)
