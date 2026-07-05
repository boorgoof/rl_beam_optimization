"""
Sensitivity analysis for ADIGE beam line parameters.

Computes the sensitivity of the score to each tunable parameter via central
finite differences, using TraceWin as the physics simulator. The values are
expressed in the same convention used by adige.py:

    sensitivity_p = TARGET_SCORE_CHANGE * 2δ / |score(p+δ) - score(p-δ)|

This is the inverse of the standard derivative (Δscore/Δparam), scaled by
TARGET_SCORE_CHANGE. It represents how much you need to move parameter p to
change the score by TARGET_SCORE_CHANGE points — the natural unit for action
step sizing in the RL environment.

TraceWin Monte Carlo noise is handled with common random numbers (CRN): the
+δ and -δ runs of each pair share the same TraceWin random_seed, so the noise
is correlated and cancels in the difference; each repeat uses a different seed
so the paired differences are independent samples. A startup probe measures
the nominal-score noise floor and checks whether TraceWin is deterministic at
fixed seed (if not, rerun with --tracewin-threads 1). A sensitivity is accepted
only when the aggregated |Δscore| clears a signal-to-noise threshold.

Run as a script:
    python -m beam_optimization.config.utility.sensitivity

The script prints a human-readable stability table and a copy-paste block ready
to replace the sensitivity= values in adige.py. It does NOT modify any files.
"""
from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

from beam_optimization.config.adige import PARAMETERS, default_params


# Sensitivity convention: parameter change that moves the score by exactly
# this many points. 1.0 = the plain inverse derivative |dp/dscore|.
TARGET_SCORE_CHANGE: float = 1.0
CHECKPOINT_VERSION: int = 2


def _set_seed(simulator, seed: int) -> None:
    """Fix the TraceWin random seed when the simulator supports it (CRN)."""
    if hasattr(simulator, "tracewin_params"):
        simulator.tracewin_params["random_seed"] = int(seed)


def probe_noise(
    simulator,
    *,
    base_seed: int = 42,
    n_probes: int = 5,
    verbose: bool = True,
) -> Optional[float]:
    """Measure the Monte Carlo noise floor and check seed determinism.

    Runs the nominal parameters twice with the SAME seed (determinism check:
    if the two scores differ, TraceWin is not reproducible at fixed seed —
    typically a multithreading effect; rerun with --tracewin-threads 1) and
    then n_probes times with DIFFERENT seeds to estimate sigma_noise, the
    standard deviation of the nominal score across noise realizations.

    Returns sigma_noise, or None if the probe was skipped/failed.
    """
    if n_probes <= 0:
        return None
    defaults = default_params()

    if verbose:
        print(f"Noise probe: determinism check (2 runs, seed={base_seed}) ...", flush=True)
    same_seed_scores: List[float] = []
    for _ in range(2):
        _set_seed(simulator, base_seed)
        res = simulator.simulate(copy.copy(defaults))
        if res.success:
            same_seed_scores.append(float(res.score_val))
    if len(same_seed_scores) == 2:
        drift = abs(same_seed_scores[0] - same_seed_scores[1])
        if drift == 0.0:
            print("  TraceWin is deterministic at fixed seed ✓ (identical scores)")
        elif drift < 0.05:
            # Residual multithreaded floating-point jitter: orders of magnitude
            # below the seed-to-seed noise, harmless for CRN cancellation.
            print(
                f"  TraceWin reproducible at fixed seed up to thread jitter "
                f"(|Δscore| = {drift:.2e}) ✓ — fine for CRN."
            )
        else:
            print(
                f"  WARNING: TraceWin is NOT reproducible at fixed seed "
                f"(|Δscore| = {drift:.4f} between identical runs).\n"
                f"  CRN cancellation will be partial — rerun with --tracewin-threads 1."
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
            scores.append(float(res.score_val))
    if len(scores) < 2:
        print("  WARNING: noise probe failed; sigma_noise unknown.")
        return None
    sigma = statistics.pstdev(scores)
    print(
        f"  nominal score = {statistics.fmean(scores):.4f} ± {sigma:.4f} "
        f"(n={len(scores)} seeds)"
    )
    return sigma


def compute_sensitivity(
    simulator,
    *,
    delta_scales: List[float] | None = None,
    repeats: int = 1,
    min_score_diff: float = 1e-9,
    snr_min: float = 3.0,
    base_seed: int = 42,
    aggregation: str = "median",
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> tuple[Dict[str, List[Optional[float]]], Dict[str, List[Optional[float]]]]:
    """Compute per-parameter sensitivity via CRN-paired central differences.

    For each parameter p and each delta_scale s, perturbs p by δ = s * sensitivity_current
    (keeping all other parameters at their nominal defaults). Each repeat i runs
    the +δ and -δ simulations with the SAME TraceWin seed (base_seed + i), so
    Monte Carlo noise cancels in the paired difference Δ_i = score(+δ) - score(-δ);
    different repeats use different seeds, giving independent samples of Δ.

        sensitivity_p = TARGET_SCORE_CHANGE * 2δ / |aggregate(Δ_i)|

    Repeating at multiple δ scales lets you verify gradient stability:
      - ratio sens(δ) / sens(δ/2) near 1.0 → gradient is linear, values are reliable
      - large ratio → δ too large (nonlinear regime) or too small (numerical noise)

    Args:
        simulator: Any object with a .simulate(params: dict) method returning a
                   result with .score_val (float) and .success (bool). If it also
                   exposes .tracewin_params (TraceWinSimulator does), the CRN
                   seed is set through it.
        delta_scales: Fractions of the current sensitivity to use as δ.
                      Defaults to [1.0, 0.5, 0.25].
        repeats: Number of CRN pairs per (parameter, scale). Use >= 3 so the
                 SNR acceptance test has a spread estimate.
        min_score_diff: Minimum accepted aggregated |Δscore|.
        snr_min: With >= 2 repeats, require |Δ| >= snr_min * std(Δ_i)/sqrt(n).
        base_seed: First TraceWin seed; repeat i uses base_seed + i.
        aggregation: "median" or "mean" aggregation of the paired differences.
        checkpoint_path: Optional JSON file updated after each completed
                         parameter/scale.
        resume: If True and checkpoint_path exists, reuse completed values.
        verbose: Print progress to stdout during the (potentially long) run.

    Returns:
        (results, snrs): both map param name → list (one entry per delta_scale).
        results holds sensitivity values, snrs the measured signal-to-noise
        ratio |Δ| / (std(Δ_i)/sqrt(n)). None = failed / below threshold.
    """
    if delta_scales is None:
        delta_scales = [1.0, 0.5, 0.25]
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if aggregation not in {"median", "mean"}:
        raise ValueError("aggregation must be 'median' or 'mean'")

    defaults = default_params()
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    results: Dict[str, List[Optional[float]]] = {}
    snrs: Dict[str, List[Optional[float]]] = {}
    if checkpoint is not None and resume:
        results, snrs = _load_checkpoint(checkpoint, delta_scales, base_seed)
    n_total = len(PARAMETERS) * len(delta_scales) * 2 * repeats
    run_count = 0

    def _checkpoint_save():
        if checkpoint is not None:
            _save_checkpoint(checkpoint, delta_scales, base_seed, results, snrs)

    for p in PARAMETERS:
        sens_list: List[Optional[float]] = list(results.get(p.name, []))
        snr_list: List[Optional[float]] = list(snrs.get(p.name, []))
        while len(snr_list) < len(sens_list):
            snr_list.append(None)

        for scale_idx, scale in enumerate(delta_scales):
            if scale_idx < len(sens_list):
                if verbose:
                    print(
                        f"  [resume] {p.name:<14}  scale={scale}  "
                        f"sens={_format_optional_sensitivity(sens_list[scale_idx])}",
                        flush=True,
                    )
                continue

            delta = scale * p.sensitivity

            params_plus  = copy.copy(defaults)
            params_minus = copy.copy(defaults)
            params_plus[p.key]  = p.default + delta
            params_minus[p.key] = p.default - delta

            diffs: List[float] = []

            for repeat_idx in range(repeats):
                seed = base_seed + repeat_idx
                _set_seed(simulator, seed)

                run_count += 1
                if verbose:
                    print(
                        f"  [{run_count:3d}/{n_total}] {p.name:<14}  +δ  "
                        f"(scale={scale}, seed={seed})",
                        flush=True,
                    )
                res_plus = simulator.simulate(params_plus)

                run_count += 1
                if verbose:
                    print(
                        f"  [{run_count:3d}/{n_total}] {p.name:<14}  -δ  "
                        f"(scale={scale}, seed={seed})",
                        flush=True,
                    )
                _set_seed(simulator, seed)
                res_minus = simulator.simulate(params_minus)

                if res_plus.success and res_minus.success:
                    diffs.append(float(res_plus.score_val) - float(res_minus.score_val))

            if not diffs:
                if verbose:
                    print(f"    WARNING: all pairs failed for {p.name} at scale={scale}")
                sens_list.append(None)
                snr_list.append(None)
                results[p.name] = sens_list
                snrs[p.name] = snr_list
                _checkpoint_save()
                continue

            diff_agg = _aggregate_scores(diffs, aggregation)
            score_diff = abs(diff_agg)
            diff_sem = (
                statistics.pstdev(diffs) / (len(diffs) ** 0.5) if len(diffs) >= 2 else 0.0
            )
            snr = score_diff / diff_sem if diff_sem > 0 else float("inf")

            too_small = score_diff < min_score_diff
            too_noisy = len(diffs) >= 2 and diff_sem > 0 and snr < snr_min
            if too_small or too_noisy:
                if verbose:
                    reason = "below min_score_diff" if too_small else f"SNR {snr:.1f} < {snr_min:g}"
                    print(
                        f"    WARNING: rejected for {p.name} at scale={scale} ({reason}; "
                        f"|Δ| = {score_diff:.4f}, σ(Δ)/√n = {diff_sem:.4f}, "
                        f"Δ_i = {[round(d, 4) for d in diffs]})"
                    )
                sens_list.append(None)
                snr_list.append(None if diff_sem == 0 else snr)
                results[p.name] = sens_list
                snrs[p.name] = snr_list
                _checkpoint_save()
                continue

            sensitivity_val = TARGET_SCORE_CHANGE * (2.0 * abs(delta)) / score_diff
            if verbose:
                deriv = score_diff / (2.0 * abs(delta))
                snr_str = "∞" if snr == float("inf") else f"{snr:.1f}"
                print(
                    f"    sens = {sensitivity_val:.6e}  "
                    f"(|Δ| = {score_diff:.4f}, σ(Δ)/√n = {diff_sem:.4f}, SNR = {snr_str}, "
                    f"Δscore/Δparam = {deriv:.4f}, "
                    f"Δ_i = {[round(d, 4) for d in diffs]})"
                )
            sens_list.append(sensitivity_val)
            snr_list.append(None if snr == float("inf") else snr)
            results[p.name] = sens_list
            snrs[p.name] = snr_list
            _checkpoint_save()

        results[p.name] = sens_list
        snrs[p.name] = snr_list

    return results, snrs


def _aggregate_scores(scores: List[float], aggregation: str) -> float:
    if aggregation == "mean":
        return statistics.fmean(scores)
    return statistics.median(scores)


def _format_optional_sensitivity(value: Optional[float]) -> str:
    return "FAILED" if value is None else f"{value:.6e}"


def _optional_float_lists(raw: dict) -> Dict[str, List[Optional[float]]]:
    return {
        name: [None if value is None else float(value) for value in values]
        for name, values in raw.items()
    }


def _load_checkpoint(
    checkpoint_path: Path,
    delta_scales: List[float],
    base_seed: int,
) -> tuple[Dict[str, List[Optional[float]]], Dict[str, List[Optional[float]]]]:
    if not checkpoint_path.exists():
        return {}, {}
    with open(checkpoint_path) as f:
        payload = json.load(f)

    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has version {payload.get('version')}, "
            f"this script writes version {CHECKPOINT_VERSION} (CRN-paired differences). "
            "Use --no-resume or a new --checkpoint."
        )
    saved_scales = [float(value) for value in payload.get("delta_scales", [])]
    if saved_scales != [float(value) for value in delta_scales]:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was created with delta_scales={saved_scales}, "
            f"but this run uses delta_scales={delta_scales}. Use --no-resume or a new --checkpoint."
        )
    if int(payload.get("base_seed", base_seed)) != int(base_seed):
        raise ValueError(
            f"Checkpoint {checkpoint_path} was created with base_seed={payload.get('base_seed')}, "
            f"but this run uses base_seed={base_seed}. Use --no-resume or a new --checkpoint."
        )

    return (
        _optional_float_lists(payload.get("results", {})),
        _optional_float_lists(payload.get("snrs", {})),
    )


def _save_checkpoint(
    checkpoint_path: Path,
    delta_scales: List[float],
    base_seed: int,
    results: Dict[str, List[Optional[float]]],
    snrs: Dict[str, List[Optional[float]]],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "base_seed": int(base_seed),
        "delta_scales": [float(value) for value in delta_scales],
        "results": results,
        "snrs": snrs,
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(checkpoint_path)


def _select_stable_sensitivity(
    vals: List[Optional[float]],
    *,
    stability_low: float,
    stability_high: float,
) -> Optional[float]:
    """Use the median of values that belong to at least one stable adjacent pair."""
    stable_indices: set[int] = set()
    for i in range(len(vals) - 1):
        a = vals[i]
        b = vals[i + 1]
        if a is None or b is None or b == 0:
            continue
        ratio = a / b
        if stability_low <= ratio <= stability_high:
            stable_indices.add(i)
            stable_indices.add(i + 1)

    if not stable_indices:
        return None

    stable_vals = [vals[i] for i in sorted(stable_indices) if vals[i] is not None]
    return statistics.median(stable_vals) if stable_vals else None


def print_sensitivity_report(
    results: Dict[str, List[Optional[float]]],
    delta_scales: List[float] | None = None,
    stability_low: float = 0.7,
    stability_high: float = 1.3,
    snrs: Dict[str, List[Optional[float]]] | None = None,
    sigma_noise: Optional[float] = None,
) -> None:
    """Print a stability table and a copy-paste block ready for adige.py.

    Table columns:
        name | default | current_sensitivity | sens(δ×…) per scale | stability | SNR

    Stability ratio = sens(δ×a) / sens(δ×b) for adjacent scales. Values near
    1.0 mean the gradient is stable and the result is reliable. A '!' flag
    marks parameters with no adjacent ratio inside [stability_low, high].
    SNR = worst measured |Δ| / (σ(Δ)/√n) among the accepted scales.

    The copy-paste block uses the median of values that belong to a stable
    adjacent-scale plateau. If no plateau exists, it keeps the old value.
    """
    if delta_scales is None:
        delta_scales = [1.0, 0.5, 0.25]
    snrs = snrs or {}

    scale_headers = [f"sens(δ×{s})" for s in delta_scales]
    col_w = 14

    # ── stability table ───────────────────────────────────────────────────────
    header = (
        f"{'Parameter':<14} {'default':>12} {'current_sens':>14}"
        + "".join(f" {h:>{col_w}}" for h in scale_headers)
        + f" {'stability':>10} {'SNR':>8}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for p in PARAMETERS:
        vals = results.get(p.name, [None] * len(delta_scales))
        sens_strs = [f"{v:.6e}" if v is not None else "    FAILED" for v in vals]

        stab_str = "         —"
        ratios = [
            vals[i] / vals[i + 1]
            for i in range(len(vals) - 1)
            if vals[i] is not None and vals[i + 1] not in (None, 0)
        ]
        if ratios:
            stable = any(stability_low <= r <= stability_high for r in ratios)
            ratio_preview = min(ratios, key=lambda r: abs(r - 1.0))
            flag = "  " if stable else " !"
            stab_str = f"{ratio_preview:>9.3f}{flag}"

        snr_vals = [
            s for s, v in zip(snrs.get(p.name, []), vals)
            if s is not None and v is not None
        ]
        snr_str = f"{min(snr_vals):>8.1f}" if snr_vals else "       —"

        row = (
            f"{p.name:<14} {p.default:>12.6g} {p.sensitivity:>14.6e}"
            + "".join(f" {s:>{col_w}}" for s in sens_strs)
            + f" {stab_str} {snr_str}"
        )
        print(row)

    print(sep)
    print(
        "\nstability = best adjacent ratio sens(δ×a) / sens(δ×b).  "
        f"Near 1.0 → stable.  '!' → no adjacent ratio inside [{stability_low}, {stability_high}]."
    )
    print("SNR = worst |Δscore| / (σ(Δ)/√n) among accepted scales (CRN-paired differences).")
    if sigma_noise is not None:
        print(f"Nominal-score noise floor (different seeds): σ = {sigma_noise:.4f}")
    print()

    # ── copy-paste block ──────────────────────────────────────────────────────
    bar = "─" * 80
    print(bar)
    print(
        "Copy-paste block for adige.py  "
        f"(sensitivity = parameter change for {TARGET_SCORE_CHANGE:g} score points; "
        "uses stable adjacent-scale plateau only):"
    )
    print(bar)
    for p in PARAMETERS:
        vals = results.get(p.name, [])
        new_sens = _select_stable_sensitivity(
            vals,
            stability_low=stability_low,
            stability_high=stability_high,
        )

        hw_min_s = "None" if p.hw_min is None else repr(p.hw_min)
        hw_max_s = "None" if p.hw_max is None else repr(p.hw_max)

        if new_sens is None:
            print(
                f"    # {p.name}: NO STABLE PLATEAU — keep old value {p.sensitivity:.15e}"
            )
            new_sens = p.sensitivity
        print(
            f'    ParameterSpec("{p.name}", "{p.key}", marker={p.marker}, '
            f'default={p.default!r}, sensitivity={new_sens:.15e}, '
            f'hw_min={hw_min_s}, hw_max={hw_max_s}, '
            f'action_scale_rl={p.action_scale_rl:.15e}, '
            f'reset_scale={p.reset_scale:.15e}),'
        )
    print(bar)


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
            f"Here sensitivity means parameter change for {TARGET_SCORE_CHANGE:g} score points. "
            "Prints a stability table and a copy-paste block — does NOT modify adige.py."
        )
    )
    parser.add_argument(
        "--ini",
        default=str(DEFAULT_TRACEWIN_INI),
        help="Path to TraceWin .ini project file (default: %(default)s)",
    )
    parser.add_argument(
        "--calc-dir",
        default=str(DEFAULT_SENSITIVITY_CALC_DIR),
        help="Temporary calculation directory for TraceWin outputs",
    )
    parser.add_argument(
        "--delta-scales",
        nargs="+",
        type=float,
        default=[50.0, 12.0, 3.0, 1.0],
        metavar="S",
        help=(
            "Multiples of the current (1-point) sensitivity used as δ. "
            "δ×S moves the score by ≈2S points; CRN keeps even small deltas "
            "above the noise (default: 50 12 3 1)"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="CRN pairs (+δ/-δ with the same seed) per parameter/scale (default: %(default)s)",
    )
    parser.add_argument(
        "--min-score-diff",
        type=float,
        default=0.5,
        help=(
            "Minimum accepted aggregated |score(+δ)-score(-δ)|. "
            "Smaller differences are treated as noise (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="First TraceWin random_seed; repeat i uses base_seed + i (default: %(default)s)",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=3.0,
        help=(
            "Reject a sensitivity when |Δ| < snr_min * σ(Δ)/√n over the CRN "
            "pairs (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--noise-probes",
        type=int,
        default=5,
        help=(
            "Nominal runs with different seeds to measure the noise floor "
            "(plus a 2-run determinism check). 0 skips the probe (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--tracewin-threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "TraceWin nbr_thread (default: all CPUs). Use 1 if the determinism "
            "probe reports non-reproducible runs at fixed seed."
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "mean"],
        default="median",
        help="How to aggregate repeated scores (default: %(default)s)",
    )
    parser.add_argument(
        "--stability-low",
        type=float,
        default=0.7,
        help="Lower accepted adjacent stability ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--stability-high",
        type=float,
        default=1.3,
        help="Upper accepted adjacent stability ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="TraceWin timeout per simulation in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_SENSITIVITY_CHECKPOINT),
        help=(
            "JSON checkpoint updated after each completed parameter/scale "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and recompute from scratch.",
    )
    args = parser.parse_args()

    print(f"TraceWin project : {args.ini}")
    print(f"Calc dir         : {args.calc_dir}")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Resume           : {not args.no_resume}")
    print(f"Delta scales     : {args.delta_scales}")
    print(f"Repeats (CRN)    : {args.repeats}")
    print(f"Base seed        : {args.base_seed}")
    print(f"SNR min          : {args.snr_min}")
    print(f"TraceWin threads : {args.tracewin_threads or 'all CPUs'}")
    print(f"Aggregation      : {args.aggregation}")
    print(f"Min score diff   : {args.min_score_diff}")
    print(f"Target Δscore    : {TARGET_SCORE_CHANGE:g}")
    print(f"Stability window : [{args.stability_low}, {args.stability_high}]")
    print(f"Parameters       : {len(PARAMETERS)}")
    print(f"Total TW runs    : {len(PARAMETERS) * len(args.delta_scales) * 2 * args.repeats}"
          f" (+ {2 + max(0, args.noise_probes)} probe runs)")
    print()

    simulator = TraceWinSimulator(
        project_file=args.ini,
        calc_dir=args.calc_dir,
        timeout=args.timeout,
        num_threads=args.tracewin_threads,
    )

    sigma_noise = probe_noise(
        simulator,
        base_seed=args.base_seed,
        n_probes=args.noise_probes,
    )

    sensitivity_results, sensitivity_snrs = compute_sensitivity(
        simulator,
        delta_scales=args.delta_scales,
        repeats=args.repeats,
        min_score_diff=args.min_score_diff,
        snr_min=args.snr_min,
        base_seed=args.base_seed,
        aggregation=args.aggregation,
        checkpoint_path=args.checkpoint,
        resume=not args.no_resume,
        verbose=True,
    )

    print_sensitivity_report(
        sensitivity_results,
        delta_scales=args.delta_scales,
        stability_low=args.stability_low,
        stability_high=args.stability_high,
        snrs=sensitivity_snrs,
        sigma_noise=sigma_noise,
    )
