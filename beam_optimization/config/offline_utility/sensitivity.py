"""Offline single-seed TraceWin sensitivity analysis for ADIGE parameters.

For each parameter, the search evaluates a common-random-number pair
``score(p + delta) - score(p - delta)`` and increases ``delta`` until the
requested score difference is reached or a hardware bound stops the search.

When a parameter's default sits on (or near) a hardware bound -- e.g.
AD.1EQ.02, whose default and upper bound are both zero -- or when TraceWin
rejects only one perturbation direction, the search falls back to a one-sided
difference against a cached nominal baseline instead of giving up on that
parameter.

If no perturbation at any tried delta ever produces a usable difference, the
search does not silently keep the old declared sensitivity: it estimates a
conservative value from the initial delta instead (status
``no_diff_estimated_from_initial_delta``), since reusing a value that may be
the reason the search failed to find any signal is rarely useful.

The resulting sensitivities are printed and saved as JSON; ``adige.py`` is
never modified automatically.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from beam_optimization.config.adige import PARAMETERS, default_params


TARGET_SCORE_CHANGE: float = 1.0


def _set_seed(simulator, seed: int) -> None:
    """Fix the TraceWin random seed when the simulator supports it."""
    if hasattr(simulator, "tracewin_params"):
        simulator.tracewin_params["random_seed"] = int(seed)


def _initial_delta(parameter) -> float:
    """Return a small, family-appropriate initial perturbation."""
    if parameter.name.startswith("AD.SO"):
        return 1e-4
    if parameter.name.startswith("AD.MS"):
        return 1e-6
    if parameter.name.startswith("AD.D"):
        return 1e-7
    if parameter.name.startswith("AD.1EQ") or parameter.name.startswith("AD.EM"):
        return 0.1
    return max(abs(float(parameter.default)), 1.0) * 1e-5


def _valid_directions(parameter, delta: float) -> tuple[str, ...]:
    """Return the perturbation directions allowed by the hardware bounds."""
    lower = parameter.hw_min
    upper = parameter.hw_max
    if parameter.name.startswith("AD.SO") and lower is None:
        lower = 0.0
    minus = parameter.default - delta
    plus = parameter.default + delta

    directions = []
    if upper is None or plus <= upper:
        directions.append("plus")
    if lower is None or minus >= lower:
        directions.append("minus")
    return tuple(directions)


def _format_delta(value: float) -> str:
    return f"{value:.3g}"


def _run_single(
    simulator,
    parameter,
    defaults: Dict[str, float],
    *,
    value: float,
    label: str,
    delta: float,
    seed: int,
    run_count: int,
    total_runs: int,
    verbose: bool,
):
    """Run one deterministic TraceWin probe and print a useful failure reason."""
    params = defaults.copy()
    params[parameter.key] = value
    _set_seed(simulator, seed)
    run_count += 1
    if verbose:
        delta_text = "nominal" if label == "baseline" else f"delta={_format_delta(delta)}"
        print(
            f"  [{run_count:4d}/{total_runs}] {parameter.name:<14} {label:<8} "
            f"({delta_text}, seed={seed})",
            flush=True,
        )
    result = simulator.simulate(params)
    if verbose and not result.success:
        error = getattr(result, "error", None)
        suffix = "" if not error else f": {str(error)[:300]}"
        print(f"    {label} failed{suffix}", flush=True)
    return run_count, result


def _run_adaptive_difference(
    simulator,
    parameter,
    defaults: Dict[str, float],
    delta: float,
    *,
    baseline,
    directions: tuple[str, ...],
    seed: int,
    run_count: int,
    total_runs: int,
    verbose: bool,
) -> tuple[int, Optional[float], Optional[str]]:
    """Measure the best available difference, falling back from central.

    When both perturbations are physically allowed they are attempted first.
    If only one succeeds -- because the other is outside the hardware bounds,
    or TraceWin rejects it at runtime -- the cached nominal baseline is used
    to compute a one-sided difference instead of giving up on this delta.
    """
    values = {"plus": parameter.default + delta, "minus": parameter.default - delta}
    labels = {"plus": "+delta", "minus": "-delta"}
    results = {}

    for direction in directions:
        run_count, results[direction] = _run_single(
            simulator,
            parameter,
            defaults,
            value=values[direction],
            label=labels[direction],
            delta=delta,
            seed=seed,
            run_count=run_count,
            total_runs=total_runs,
            verbose=verbose,
        )

    plus = results.get("plus")
    minus = results.get("minus")
    if plus is not None and plus.success and minus is not None and minus.success:
        return (
            run_count,
            float(plus.score_val) - float(minus.score_val),
            "central",
        )

    successful_direction = None
    if plus is not None and plus.success:
        successful_direction = "plus"
    elif minus is not None and minus.success:
        successful_direction = "minus"

    if successful_direction is None:
        return run_count, None, None

    side = results[successful_direction]
    diff = float(side.score_val) - float(baseline.score_val)
    scheme = "forward" if successful_direction == "plus" else "backward"
    if verbose and len(directions) == 2:
        print(
            f"    central difference unavailable; falling back to {scheme}",
            flush=True,
        )
    return run_count, diff, scheme


def compute_sensitivity(
    simulator,
    *,
    seed: int = 42,
    escalation_factor: float = 3.0,
    max_iterations: int = 8,
    target_score_diff: float = 1.0,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Compute one fixed sensitivity per parameter using a single seed."""
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

    for parameter in PARAMETERS:
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
    last_good_delta: Optional[float] = None
    best_delta: Optional[float] = None
    best_diff: Optional[float] = None
    best_scheme: Optional[str] = None
    best_iteration: Optional[int] = None
    hardware_limited = False

    if verbose:
        print(
            f"\n{parameter.name}: starting delta = {_format_delta(delta)}",
            flush=True,
        )

    run_count, baseline = _run_single(
        simulator,
        parameter,
        defaults,
        value=parameter.default,
        label="baseline",
        delta=0.0,
        seed=seed,
        run_count=run_count,
        total_runs=total_runs,
        verbose=verbose,
    )
    if not baseline.success:
        if verbose:
            print(
                f"    baseline failed for {parameter.name}; cannot measure sensitivity",
                flush=True,
            )
        return run_count, {
            "sensitivity": parameter.sensitivity,
            "delta": None,
            "diff": None,
            "scheme": None,
            "iterations": 0,
            "status": "baseline_failed",
        }

    for iteration in range(1, max_iterations + 1):
        directions = _valid_directions(parameter, delta)
        if not directions:
            hardware_limited = True
            if verbose:
                print(
                    f"    delta={_format_delta(delta)} exceeds both available "
                    "hardware directions; stopping",
                    flush=True,
                )
            break

        run_count, diff, scheme = _run_adaptive_difference(
            simulator,
            parameter,
            defaults,
            delta,
            baseline=baseline,
            directions=directions,
            seed=seed,
            run_count=run_count,
            total_runs=total_runs,
            verbose=verbose,
        )

        if diff is None:
            if verbose:
                print(
                    f"    delta={_format_delta(delta)}: TraceWin failure; backing off",
                    flush=True,
                )
            delta = (
                delta / escalation_factor
                if last_good_delta is None
                else (delta + last_good_delta) / 2.0
            )
            continue

        if best_diff is None or abs(diff) > abs(best_diff):
            best_delta = delta
            best_diff = diff
            best_scheme = scheme
            best_iteration = iteration

        if abs(diff) >= target_score_diff:
            span = 2.0 * delta if scheme == "central" else delta
            sensitivity = TARGET_SCORE_CHANGE * span / abs(diff)
            if verbose:
                print(
                    f"    ACCEPTED {scheme}: delta={_format_delta(delta)} diff={diff:.4f} "
                    f"sensitivity={sensitivity:.6e}",
                    flush=True,
                )
            return run_count, {
                "sensitivity": sensitivity,
                "delta": delta,
                "diff": diff,
                "scheme": scheme,
                "iterations": iteration,
                "status": "ok",
            }

        if verbose:
            print(
                f"    {scheme}: diff={diff:.4f} below target "
                f"{target_score_diff:g}; escalating",
                flush=True,
            )
        last_good_delta = delta
        delta *= escalation_factor

    if best_diff is None or best_delta is None:
        # No perturbation ever produced a usable finite difference, even
        # after backing off/escalating delta from the initial guess. Rather
        # than silently keeping the old (possibly wrong -- and possibly the
        # reason the search never got a signal) declared sensitivity,
        # estimate a conservative value from the initial delta that was
        # actually tried first: treat it as if it alone would span
        # target_score_diff points, which gives a small/safe sensitivity.
        initial_delta = _initial_delta(parameter)
        estimated_sensitivity = TARGET_SCORE_CHANGE * initial_delta / target_score_diff
        if verbose:
            print(
                f"    WARNING: no valid measurement for {parameter.name}; "
                f"estimating sensitivity={estimated_sensitivity:.6e} from the initial delta",
                flush=True,
            )
        return run_count, {
            "sensitivity": estimated_sensitivity,
            "delta": initial_delta,
            "diff": None,
            "scheme": None,
            "iterations": max_iterations,
            "status": "no_diff_estimated_from_initial_delta",
        }

    effective_diff = max(abs(best_diff), target_score_diff)
    span = 2.0 * best_delta if best_scheme == "central" else best_delta
    sensitivity = TARGET_SCORE_CHANGE * span / effective_diff
    status = "hardware_limited" if hardware_limited else "weak_signal"
    if verbose:
        print(
            f"    SELECTED {best_scheme}: delta={_format_delta(best_delta)} "
            f"sensitivity={sensitivity:.6e} status={status}",
            flush=True,
        )
    return run_count, {
        "sensitivity": sensitivity,
        "delta": best_delta,
        "diff": best_diff,
        "scheme": best_scheme,
        "iterations": best_iteration,
        "status": status,
    }


def estimate_total_runs(max_iterations: int) -> int:
    """Upper bound assuming every parameter uses every iteration."""
    # One baseline per parameter, then at most two directions per iteration.
    return len(PARAMETERS) * (1 + max_iterations * 2)


def print_sensitivity_report(records: Dict[str, dict]) -> None:
    """Print one fixed sensitivity value per parameter."""
    header = (
        f"{'Parameter':<14} {'default':>12} {'old_sens':>12} {'new_sens':>12} "
        f"{'delta':>12} {'diff':>10} {'scheme':>10} {'iters':>6} {'status':>34}"
    )
    separator = "=" * len(header)
    print(f"\n{separator}")
    print(header)
    print(separator)

    for parameter in PARAMETERS:
        record = records[parameter.name]
        delta = record.get("delta")
        diff = record.get("diff")
        iterations = record.get("iterations")
        print(
            f"{parameter.name:<14} {parameter.default:>12.6g} "
            f"{parameter.sensitivity:>12.4e} {record['sensitivity']:>12.4e} "
            f"{('-' if delta is None else _format_delta(delta)):>12} "
            f"{('-' if diff is None else f'{diff:.4f}'):>10} "
            f"{(record.get('scheme') or '-'):>10} "
            f"{('-' if iterations is None else str(iterations)):>6} "
            f"{record['status']:>34}"
        )
    print(separator)


def save_sensitivity_json(
    records: Dict[str, dict],
    output: str | Path,
    *,
    run_config: dict,
) -> Path:
    """Save measured and previous sensitivity values in a reusable JSON report."""
    parameters = {}
    for parameter in PARAMETERS:
        parameters[parameter.name] = {
            "key": parameter.key,
            "default": parameter.default,
            "old_sensitivity": parameter.sensitivity,
            **records[parameter.name],
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "single_seed_finite_difference_with_one_sided_fallback",
        "target_score_change": TARGET_SCORE_CHANGE,
        "run_config": run_config,
        "new_sensitivity": {
            name: record["sensitivity"] for name, record in records.items()
        },
        "parameters": parameters,
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    import argparse

    from beam_optimization.config.paths import (
        DEFAULT_SENSITIVITY_OUTPUT,
        DEFAULT_TRACEWIN_INI,
        resolve_tracewin_project,
    )
    from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import (
        TraceWinSimulator,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Single-seed ADIGE sensitivity from TraceWin finite differences, "
            "with automatic one-sided fallback at hardware bounds and when "
            "one TraceWin perturbation direction fails. Prints and saves one "
            "value per parameter without modifying adige.py."
        )
    )
    tracewin_source = parser.add_mutually_exclusive_group()
    tracewin_source.add_argument(
        "--workspace",
        default=None,
        metavar="PATH",
        help="TraceWin workspace directory. Mutually exclusive with --tracewin.",
    )
    tracewin_source.add_argument(
        "--tracewin",
        default=None,
        metavar="INI",
        help=(
            "TraceWin project .ini to use. Mutually exclusive with "
            f"--workspace. Default: {DEFAULT_TRACEWIN_INI}"
        ),
    )
    parser.add_argument(
        "--calc-dir",
        default=None,
        metavar="PATH",
        help="TraceWin calculation directory (default: sensitivity_calc inside the workspace).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SENSITIVITY_OUTPUT),
        metavar="JSON",
        help="Result JSON path (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--escalation-factor", type=float, default=3.0)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--target-score-diff", type=float, default=1.0)
    parser.add_argument("--tracewin-threads", type=int, default=None, metavar="N")
    parser.add_argument("--tracewin-particles", type=int, default=10000)
    parser.add_argument("--tracewin-particle-key", default="nbr_part1")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    try:
        workspace_path, project_file = resolve_tracewin_project(
            workspace=args.workspace,
            tracewin=args.tracewin,
        )
    except ValueError as exc:
        parser.error(str(exc))

    calc_dir = Path(args.calc_dir) if args.calc_dir else workspace_path / "sensitivity_calc"

    tracewin_params = {args.tracewin_particle_key: int(args.tracewin_particles)}
    total_runs = estimate_total_runs(args.max_iterations)

    print(f"TraceWin workspace    : {workspace_path}")
    print(f"TraceWin project      : {project_file}")
    print(f"Calc dir              : {calc_dir}")
    print(f"JSON output           : {args.output}")
    print(f"Seed                  : {args.seed}")
    print(f"Escalation factor     : {args.escalation_factor}")
    print(f"Max iterations        : {args.max_iterations}")
    print(f"Target score diff     : {args.target_score_diff}")
    print(f"TraceWin threads      : {args.tracewin_threads or 'all CPUs'}")
    print(f"TraceWin particles    : {args.tracewin_particles}")
    print(f"Parameters            : {len(PARAMETERS)}")
    print(f"Total TW runs         : up to {total_runs} (worst case)")

    simulator = TraceWinSimulator(
        project_file=str(project_file),
        calc_dir=str(calc_dir),
        timeout=args.timeout,
        tracewin_params=tracewin_params,
        num_threads=args.tracewin_threads,
        initial_npart=args.tracewin_particles,
    )
    records = compute_sensitivity(
        simulator,
        seed=args.seed,
        escalation_factor=args.escalation_factor,
        max_iterations=args.max_iterations,
        target_score_diff=args.target_score_diff,
        verbose=True,
    )
    print_sensitivity_report(records)

    output = save_sensitivity_json(
        records,
        args.output,
        run_config={
            "project": str(Path(args.ini).expanduser().resolve()),
            "calc_dir": str(Path(args.calc_dir).expanduser().resolve()),
            "seed": args.seed,
            "escalation_factor": args.escalation_factor,
            "max_iterations": args.max_iterations,
            "target_score_diff": args.target_score_diff,
            "tracewin_threads": args.tracewin_threads,
            "tracewin_particles": args.tracewin_particles,
            "tracewin_particle_key": args.tracewin_particle_key,
            "timeout": args.timeout,
        },
    )
    print(f"JSON saved to: {output}")


if __name__ == "__main__":
    main()
