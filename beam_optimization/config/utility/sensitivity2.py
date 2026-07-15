"""
Quick single-seed sensitivity analysis for ADIGE beam line parameters.

Companion to sensitivity.py, for when a full multi-seed run (repeats=10,
common random numbers averaged across seeds) is too slow. This version uses
a single fixed seed per parameter — no averaging, no dispersion/stability
check — and simply finds the smallest delta whose paired difference

    diff = score(p+delta) - score(p-delta)     [same seed for both]

reaches --target-score-diff (default 1.0), then reports:

    sensitivity_p = TARGET_SCORE_CHANGE * 2*delta / |diff|

Delta starts small (family-appropriate, same as sensitivity.py) and
escalates (`delta *= escalation_factor`) until the signal is big enough,
respecting the parameter's hardware bounds. Always returns a numeric
sensitivity per parameter, with a status describing how it was obtained
("ok", "weak_signal", "hardware_limited", or "fallback_old_value" if every
simulation failed).

Because there is only one seed, there is no way to assess measurement
stability — use sensitivity.py for a more reliable (but much slower) result
when time allows.

Run as a script:
    python -m beam_optimization.config.utility.sensitivity2

Prints one table and does not modify any files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from beam_optimization.config.adige import PARAMETERS, default_params
from beam_optimization.config.utility.sensitivity import (
    TARGET_SCORE_CHANGE,
    _format_delta,
    _initial_delta,
    _is_valid_central_delta,
    _run_pair,
)


def compute_sensitivity_single_seed(
    simulator,
    *,
    seed: int = 42,
    escalation_factor: float = 3.0,
    max_iterations: int = 8,
    target_score_diff: float = 1.0,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Compute one fixed sensitivity value per parameter, using a single seed.

    Much faster than compute_sensitivity() (no repeats to average), at the
    cost of being unable to assess how stable the estimate is. Always
    returns a numeric sensitivity per parameter.
    """
    if escalation_factor <= 1.0:
        raise ValueError(f"escalation_factor must be > 1.0, got {escalation_factor}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if target_score_diff <= 0:
        raise ValueError(f"target_score_diff must be > 0, got {target_score_diff}")

    defaults = default_params()
    n_total = _estimate_total_runs_single_seed(max_iterations)
    run_count = 0
    records: Dict[str, dict] = {}

    for p in PARAMETERS:
        run_count, record = _search_single_seed_for_param(
            simulator,
            p,
            defaults,
            seed=seed,
            escalation_factor=escalation_factor,
            max_iterations=max_iterations,
            target_score_diff=target_score_diff,
            run_count=run_count,
            n_total=n_total,
            verbose=verbose,
        )
        records[p.name] = record

    return records


def _search_single_seed_for_param(
    simulator,
    p,
    defaults: Dict[str, float],
    *,
    seed: int,
    escalation_factor: float,
    max_iterations: int,
    target_score_diff: float,
    run_count: int,
    n_total: int,
    verbose: bool,
) -> tuple[int, dict]:
    delta = _initial_delta(p)
    last_good_delta: Optional[float] = None
    best_delta: Optional[float] = None
    best_diff: Optional[float] = None
    hw_limited = False

    if verbose:
        print(f"\n{p.name}: starting delta = {_format_delta(delta)}", flush=True)

    for _iteration in range(1, max_iterations + 1):
        if not _is_valid_central_delta(p, delta):
            hw_limited = True
            if verbose:
                print(f"    delta={_format_delta(delta)} would exceed hardware bounds; stopping")
            break

        run_count, diff = _run_pair(
            simulator,
            p,
            defaults,
            delta,
            seed=seed,
            run_count=run_count,
            n_total=n_total,
            label="single",
            verbose=verbose,
        )

        if diff is None:
            if verbose:
                print(f"    delta={_format_delta(delta)}: TraceWin failure; backing off")
            delta = delta / escalation_factor if last_good_delta is None else (delta + last_good_delta) / 2.0
            continue

        if best_diff is None or abs(diff) > abs(best_diff):
            best_delta, best_diff = delta, diff

        if abs(diff) >= target_score_diff:
            sensitivity = TARGET_SCORE_CHANGE * 2.0 * abs(delta) / abs(diff)
            if verbose:
                print(f"    ACCEPTED delta={_format_delta(delta)} diff={diff:.4f} sensitivity={sensitivity:.6e}")
            return run_count, {
                "sensitivity": sensitivity,
                "delta": delta,
                "diff": diff,
                "iterations": _iteration,
                "status": "ok",
            }

        if verbose:
            print(f"    delta={_format_delta(delta)} diff={diff:.4f} below target {target_score_diff:g}; escalating")
        last_good_delta = delta
        delta *= escalation_factor

    if best_diff is None:
        if verbose:
            print(f"    WARNING: no valid measurement for {p.name}; keeping old sensitivity")
        return run_count, {
            "sensitivity": p.sensitivity,
            "delta": None,
            "diff": None,
            "iterations": max_iterations,
            "status": "fallback_old_value",
        }

    effective_diff = max(abs(best_diff), target_score_diff)
    sensitivity = TARGET_SCORE_CHANGE * 2.0 * abs(best_delta) / effective_diff
    status = "hardware_limited" if hw_limited else "weak_signal"
    if verbose:
        print(f"    SELECTED delta={_format_delta(best_delta)} sensitivity={sensitivity:.6e} status={status}")

    return run_count, {
        "sensitivity": sensitivity,
        "delta": best_delta,
        "diff": best_diff,
        "iterations": max_iterations,
        "status": status,
    }


def _estimate_total_runs_single_seed(max_iterations: int) -> int:
    """Upper bound on TraceWin runs, assuming every parameter uses all iterations."""
    return len(PARAMETERS) * max_iterations * 2


def print_sensitivity2_report(records: Dict[str, dict]) -> None:
    """Print a single table with one fixed sensitivity value per parameter."""
    header = (
        f"{'Parameter':<14} {'default':>12} {'old_sens':>12} {'new_sens':>12} "
        f"{'delta':>12} {'diff':>10} {'iters':>6} {'status':>18}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for p in PARAMETERS:
        record = records.get(p.name, {})
        new_sens = record.get("sensitivity")
        delta = record.get("delta")
        diff = record.get("diff")
        iterations = record.get("iterations")
        status = record.get("status", "n/a")

        new_sens_str = "-" if new_sens is None else f"{new_sens:.4e}"
        delta_str = "-" if delta is None else _format_delta(delta)
        diff_str = "-" if diff is None else f"{diff:.4f}"
        iters_str = "-" if iterations is None else str(iterations)

        row = (
            f"{p.name:<14} {p.default:>12.6g} {p.sensitivity:>12.4e} {new_sens_str:>12} "
            f"{delta_str:>12} {diff_str:>10} {iters_str:>6} {status:>18}"
        )
        print(row)

    print(sep)
    print(
        "status: ok = target score diff reached; weak_signal = escalated to "
        "max_iterations without reaching the target; fallback_old_value = every "
        "simulation failed, kept the adige.py value; hardware_limited = hit the "
        "hardware bound before reaching the target."
    )
    print(
        "Single seed, no averaging: use sensitivity.py for a more reliable "
        "(but slower) measurement."
    )


if __name__ == "__main__":
    import argparse

    from beam_optimization.config.paths import (
        DEFAULT_SENSITIVITY_CALC_DIR,
        DEFAULT_TRACEWIN_INI,
    )
    from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import (
        TraceWinSimulator,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Quick single-seed ADIGE parameter sensitivity from TraceWin finite "
            f"differences. Sensitivity means parameter change for {TARGET_SCORE_CHANGE:g} "
            "score point. Always prints one fixed sensitivity value per parameter. "
            "Does NOT modify adige.py."
        )
    )
    parser.add_argument("--ini", default=str(DEFAULT_TRACEWIN_INI))
    parser.add_argument("--calc-dir", default=str(DEFAULT_SENSITIVITY_CALC_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--escalation-factor", type=float, default=3.0,
        help="Multiply delta by this factor when the signal is still below the target (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=8,
        help="Max escalation steps per parameter before giving up (default: %(default)s)",
    )
    parser.add_argument(
        "--target-score-diff", type=float, default=1.0,
        help="Stop escalating once |diff| reaches this many score points (default: %(default)s)",
    )
    parser.add_argument("--tracewin-threads", type=int, default=None, metavar="N")
    parser.add_argument(
        "--tracewin-particles", type=int, default=10000,
        help="TraceWin macro-particles for sensitivity only (default: %(default)s)",
    )
    parser.add_argument(
        "--tracewin-particle-key", default="nbr_part1",
        help="TraceWin CLI key used to override particle count (default: %(default)s)",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    tracewin_params = {args.tracewin_particle_key: int(args.tracewin_particles)}

    total_runs = _estimate_total_runs_single_seed(args.max_iterations)

    print(f"TraceWin project      : {args.ini}")
    print(f"Calc dir              : {args.calc_dir}")
    print(f"Seed                  : {args.seed}")
    print(f"Escalation factor     : {args.escalation_factor}")
    print(f"Max iterations        : {args.max_iterations}")
    print(f"Target score diff     : {args.target_score_diff}")
    print(f"TraceWin threads      : {args.tracewin_threads or 'all CPUs'}")
    print(f"TraceWin particles    : {args.tracewin_particles}")
    print(f"TraceWin particle key : {args.tracewin_particle_key}")
    print(f"Target score change   : {TARGET_SCORE_CHANGE:g}")
    print(f"Parameters            : {len(PARAMETERS)}")
    print(f"Total TW runs         : up to {total_runs} (worst case)")
    print()

    simulator = TraceWinSimulator(
        project_file=args.ini,
        calc_dir=args.calc_dir,
        timeout=args.timeout,
        tracewin_params=tracewin_params,
        num_threads=args.tracewin_threads,
        initial_npart=args.tracewin_particles,
    )

    records = compute_sensitivity_single_seed(
        simulator,
        seed=args.seed,
        escalation_factor=args.escalation_factor,
        max_iterations=args.max_iterations,
        target_score_diff=args.target_score_diff,
        verbose=True,
    )

    print_sensitivity2_report(records)
