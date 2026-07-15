"""Targeted TraceWin sensitivity search for AD.1EQ.01 and AD.1EQ.02.

Unlike sensitivity2, this search supports one-sided finite differences when a
parameter's nominal value lies on a hardware bound.  This is required for
AD.1EQ.02, whose default and upper bound are both zero.
"""
from __future__ import annotations

from typing import Dict, Optional

from beam_optimization.config.adige import PARAMETERS, default_params
from beam_optimization.config.utility.sensitivity import (
    TARGET_SCORE_CHANGE,
    _format_delta,
    _initial_delta,
    _set_seed,
)


TARGET_PARAMETER_NAMES = ("AD.1EQ.01", "AD.1EQ.02")
TARGET_PARAMETERS = tuple(p for p in PARAMETERS if p.name in TARGET_PARAMETER_NAMES)

if tuple(p.name for p in TARGET_PARAMETERS) != TARGET_PARAMETER_NAMES:
    raise RuntimeError(
        "Could not find the expected sensitivity3 parameters in adige.PARAMETERS: "
        f"{TARGET_PARAMETER_NAMES}"
    )


def compute_sensitivity_eq_pair(
    simulator,
    *,
    seed: int = 42,
    escalation_factor: float = 3.0,
    max_iterations: int = 12,
    target_score_diff: float = 1.0,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Estimate sensitivity for AD.1EQ.01 and AD.1EQ.02 with one fixed seed."""
    if escalation_factor <= 1.0:
        raise ValueError(f"escalation_factor must be > 1.0, got {escalation_factor}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if target_score_diff <= 0:
        raise ValueError(f"target_score_diff must be > 0, got {target_score_diff}")

    defaults = default_params()
    total_runs = estimate_total_runs(max_iterations)
    run_count = 0
    records: Dict[str, dict] = {}

    for parameter in TARGET_PARAMETERS:
        run_count, record = _search_parameter(
            simulator,
            parameter,
            defaults,
            seed=seed,
            escalation_factor=escalation_factor,
            max_iterations=max_iterations,
            target_score_diff=target_score_diff,
            run_count=run_count,
            total_runs=total_runs,
            verbose=verbose,
        )
        records[parameter.name] = record

    return records


def _search_parameter(
    simulator,
    parameter,
    defaults: Dict[str, float],
    *,
    seed: int,
    escalation_factor: float,
    max_iterations: int,
    target_score_diff: float,
    run_count: int,
    total_runs: int,
    verbose: bool,
) -> tuple[int, dict]:
    delta = _initial_delta(parameter)
    best: Optional[dict] = None
    hardware_limited = False

    if verbose:
        print(
            f"\n{parameter.name}: starting delta = {_format_delta(delta)}",
            flush=True,
        )

    for iteration in range(1, max_iterations + 1):
        scheme = _difference_scheme(parameter, delta)
        if scheme is None:
            hardware_limited = True
            if verbose:
                print(
                    f"    delta={_format_delta(delta)} exceeds both available "
                    "hardware directions; stopping",
                    flush=True,
                )
            break

        run_count, diff = _run_difference(
            simulator,
            parameter,
            defaults,
            delta,
            scheme=scheme,
            seed=seed,
            run_count=run_count,
            total_runs=total_runs,
            verbose=verbose,
        )

        if diff is None:
            if verbose:
                print("    TraceWin failure; trying a smaller delta", flush=True)
            delta /= escalation_factor
            continue

        span = 2.0 * delta if scheme == "central" else delta
        candidate = {
            "delta": delta,
            "diff": diff,
            "span": span,
            "scheme": scheme,
            "iterations": iteration,
        }
        if best is None or abs(diff) > abs(best["diff"]):
            best = candidate

        if abs(diff) >= target_score_diff:
            candidate["sensitivity"] = (
                TARGET_SCORE_CHANGE * span / abs(diff)
            )
            candidate["status"] = "ok"
            if verbose:
                print(
                    f"    ACCEPTED {scheme}: delta={_format_delta(delta)} "
                    f"diff={diff:.6g} sensitivity={candidate['sensitivity']:.6e}",
                    flush=True,
                )
            candidate.pop("span")
            return run_count, candidate

        if verbose:
            print(
                f"    {scheme}: diff={diff:.6g} below target "
                f"{target_score_diff:g}; escalating",
                flush=True,
            )
        delta *= escalation_factor

    if best is None:
        return run_count, {
            "sensitivity": parameter.sensitivity,
            "delta": None,
            "diff": None,
            "scheme": None,
            "iterations": max_iterations,
            "status": "fallback_old_value",
        }

    best["sensitivity"] = (
        TARGET_SCORE_CHANGE
        * best["span"]
        / max(abs(best["diff"]), target_score_diff)
    )
    best["status"] = "hardware_limited" if hardware_limited else "weak_signal"
    best.pop("span")
    return run_count, best


def _difference_scheme(parameter, delta: float) -> Optional[str]:
    """Choose central, forward, or backward differences within hardware bounds."""
    minus = parameter.default - delta
    plus = parameter.default + delta
    minus_valid = parameter.hw_min is None or minus >= parameter.hw_min
    plus_valid = parameter.hw_max is None or plus <= parameter.hw_max

    if minus_valid and plus_valid:
        return "central"
    if plus_valid:
        return "forward"
    if minus_valid:
        return "backward"
    return None


def _run_difference(
    simulator,
    parameter,
    defaults: Dict[str, float],
    delta: float,
    *,
    scheme: str,
    seed: int,
    run_count: int,
    total_runs: int,
    verbose: bool,
) -> tuple[int, Optional[float]]:
    if scheme == "central":
        first_value = parameter.default + delta
        second_value = parameter.default - delta
        first_label, second_label = "+delta", "-delta"
    elif scheme == "forward":
        first_value = parameter.default + delta
        second_value = parameter.default
        first_label, second_label = "+delta", "baseline"
    elif scheme == "backward":
        first_value = parameter.default - delta
        second_value = parameter.default
        first_label, second_label = "-delta", "baseline"
    else:
        raise ValueError(f"Unknown finite-difference scheme: {scheme}")

    results = []
    for value, label in ((first_value, first_label), (second_value, second_label)):
        params = defaults.copy()
        params[parameter.key] = value
        _set_seed(simulator, seed)
        run_count += 1
        if verbose:
            print(
                f"  [{run_count:3d}/{total_runs}] {parameter.name:<10} {label:<8} "
                f"({scheme}, delta={_format_delta(delta)}, seed={seed})",
                flush=True,
            )
        results.append(simulator.simulate(params))

    if not all(result.success for result in results):
        return run_count, None
    return run_count, float(results[0].score_val) - float(results[1].score_val)


def estimate_total_runs(max_iterations: int) -> int:
    return len(TARGET_PARAMETERS) * max_iterations * 2


def print_sensitivity3_report(records: Dict[str, dict]) -> None:
    header = (
        f"{'Parameter':<12} {'old_sens':>12} {'new_sens':>12} "
        f"{'delta':>12} {'diff':>12} {'scheme':>10} {'status':>20}"
    )
    print(f"\n{'=' * len(header)}")
    print(header)
    print("=" * len(header))
    for parameter in TARGET_PARAMETERS:
        record = records[parameter.name]
        delta = record.get("delta")
        diff = record.get("diff")
        print(
            f"{parameter.name:<12} {parameter.sensitivity:>12.4e} "
            f"{record['sensitivity']:>12.4e} "
            f"{('-' if delta is None else _format_delta(delta)):>12} "
            f"{('-' if diff is None else f'{diff:.6g}'):>12} "
            f"{(record.get('scheme') or '-'):>10} {record['status']:>20}"
        )
    print("=" * len(header))


def main() -> None:
    import argparse

    from beam_optimization.config.paths import (
        DEFAULT_TRACEWIN_INI,
        new_tracewin_env_calc_dir,
    )
    from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import (
        TraceWinSimulator,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Targeted single-seed TraceWin sensitivity for AD.1EQ.01 and "
            "AD.1EQ.02, with automatic one-sided differences at hardware bounds."
        )
    )
    parser.add_argument("--ini", default=str(DEFAULT_TRACEWIN_INI))
    parser.add_argument(
        "--calc-dir",
        default=None,
        help="TraceWin calculation directory (default: unique directory under /tmp)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--escalation-factor", type=float, default=3.0)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--target-score-diff", type=float, default=1.0)
    parser.add_argument("--tracewin-threads", type=int, default=None, metavar="N")
    parser.add_argument("--tracewin-particles", type=int, default=10000)
    parser.add_argument("--tracewin-particle-key", default="nbr_part1")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    calc_dir = args.calc_dir or str(new_tracewin_env_calc_dir())
    tracewin_params = {args.tracewin_particle_key: int(args.tracewin_particles)}

    print(f"TraceWin project      : {args.ini}")
    print(f"Calc dir              : {calc_dir}")
    print(f"Parameters            : {', '.join(TARGET_PARAMETER_NAMES)}")
    print(f"Seed                  : {args.seed}")
    print(f"Escalation factor     : {args.escalation_factor}")
    print(f"Max iterations        : {args.max_iterations}")
    print(f"Target score diff     : {args.target_score_diff}")
    print(f"TraceWin threads      : {args.tracewin_threads or 'all CPUs'}")
    print(f"TraceWin particles    : {args.tracewin_particles}")
    print(f"Total TW runs         : up to {estimate_total_runs(args.max_iterations)}")

    simulator = TraceWinSimulator(
        project_file=args.ini,
        calc_dir=calc_dir,
        timeout=args.timeout,
        tracewin_params=tracewin_params,
        num_threads=args.tracewin_threads,
        initial_npart=args.tracewin_particles,
    )
    records = compute_sensitivity_eq_pair(
        simulator,
        seed=args.seed,
        escalation_factor=args.escalation_factor,
        max_iterations=args.max_iterations,
        target_score_diff=args.target_score_diff,
        verbose=True,
    )
    print_sensitivity3_report(records)


if __name__ == "__main__":
    main()
