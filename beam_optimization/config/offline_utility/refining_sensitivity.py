"""Refine the currently declared ADIGE parameter sensitivities with TraceWin.

For every parameter this command starts from its current ``ParameterSpec``
sensitivity and searches for a physical interval whose absolute score change
is ``target_score_diff`` (one point by default).  A different random TraceWin
seed is drawn for every parameter and then kept fixed for that parameter's
baseline and all of its probes, so Monte Carlo noise does not masquerade as a
parameter effect.

The command only reports proposed values.  It never edits ``adige.py``.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from beam_optimization.config.adige import PARAMETERS, default_params
from beam_optimization.config.offline_utility.sensitivity import (
    _direction_limit,
    _format_delta,
    _run_single,
)


DEFAULT_TARGET_SCORE_DIFF = 1.0
DEFAULT_TOLERANCE = 0.10
DEFAULT_MAX_BASELINE_ATTEMPTS = 5
MAX_PROPORTIONAL_STEP = 3.0


def _initial_span(parameter) -> float:
    span = abs(float(parameter.sensitivity))
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError(
            f"{parameter.name} has invalid sensitivity {parameter.sensitivity!r}"
        )
    return span


def _allocate_span(parameter, requested_span: float) -> tuple[float, float]:
    """Split a total interval around the default, respecting hardware bounds."""
    limits = {
        "plus": _direction_limit(parameter, "plus"),
        "minus": _direction_limit(parameter, "minus"),
    }
    half = requested_span / 2.0
    allocated = {
        direction: min(half, limits[direction])
        for direction in ("plus", "minus")
    }
    remaining = max(0.0, requested_span - sum(allocated.values()))

    # If one side is clipped, place the unused part on the other side.  This
    # retains the requested total secant span whenever hardware permits it.
    for direction in ("plus", "minus"):
        capacity = limits[direction] - allocated[direction]
        addition = min(remaining, capacity)
        allocated[direction] += addition
        remaining -= addition
        if remaining <= 0.0:
            break
    return float(allocated["plus"]), float(allocated["minus"])


def _geometric_midpoint(lower: float, upper: float) -> float:
    if lower > 0.0 and upper > 0.0:
        return math.sqrt(lower * upper)
    return (lower + upper) / 2.0


def _next_span(
    *,
    current_span: float,
    score_magnitude: Optional[float],
    target: float,
    lower_span: Optional[float],
    upper_span: Optional[float],
) -> float:
    """Choose a proportional step, switching to a bracketed search when possible."""
    if lower_span is not None and upper_span is not None:
        return _geometric_midpoint(lower_span, upper_span)

    if score_magnitude is None:
        return current_span / 2.0
    if score_magnitude <= 0.0:
        return current_span * MAX_PROPORTIONAL_STEP

    proportional = current_span * target / score_magnitude
    minimum = current_span / MAX_PROPORTIONAL_STEP
    maximum = current_span * MAX_PROPORTIONAL_STEP
    return min(max(proportional, minimum), maximum)


def _draw_seed(rng: np.random.Generator, used_seeds: set) -> int:
    """Draw a fresh seed not yet used by any parameter or baseline retry."""
    while True:
        candidate = int(rng.integers(0, 2**31))
        if candidate not in used_seeds:
            used_seeds.add(candidate)
            return candidate


def _fallback_record(parameter, seed: int, status: str, iterations: int) -> dict:
    return {
        "sensitivity": abs(float(parameter.sensitivity)),
        "initial_sensitivity": abs(float(parameter.sensitivity)),
        "tested_span": None,
        "delta_plus": None,
        "delta_minus": None,
        "score_diff": None,
        "absolute_score_diff": None,
        "relative_target_error": None,
        "scheme": None,
        "iterations": iterations,
        "seed": seed,
        "measured": False,
        "converged": False,
        "status": status,
    }


def _refine_parameter(
    simulator,
    parameter,
    defaults: Dict[str, float],
    *,
    rng: np.random.Generator,
    used_seeds: set,
    target_score_diff: float,
    tolerance: float,
    max_iterations: int,
    max_baseline_attempts: int,
    run_count: int,
    total_runs: int,
    verbose: bool,
) -> tuple[int, dict]:
    initial_span = _initial_span(parameter)
    requested_span = initial_span
    lower_span: Optional[float] = None
    upper_span: Optional[float] = None
    best: Optional[dict] = None
    last_signature: Optional[tuple[float, float]] = None

    # A baseline failure (e.g. an unlucky Monte Carlo draw that loses the
    # whole beam) does not mean the parameter itself is bad: redraw the seed
    # and retry before giving up, since the physical values are unchanged.
    seed: Optional[int] = None
    baseline = None
    for attempt in range(1, max_baseline_attempts + 1):
        seed = _draw_seed(rng, used_seeds)
        if verbose:
            print(
                f"\n{parameter.name}: baseline attempt {attempt}/{max_baseline_attempts} "
                f"seed={seed} initial total span={_format_delta(initial_span)}",
                flush=True,
            )
        run_count, baseline = _run_single(
            simulator,
            parameter,
            defaults,
            value=float(parameter.default),
            label="baseline",
            delta=0.0,
            seed=seed,
            run_count=run_count,
            total_runs=total_runs,
            verbose=verbose,
        )
        if baseline.success:
            break

    if not baseline.success:
        return run_count, _fallback_record(parameter, seed, "baseline_failed", 0)

    cache: dict[tuple[str, float], object] = {}
    iterations_done = 0
    stop_reason = "max_iterations"

    for iteration in range(1, max_iterations + 1):
        iterations_done = iteration
        delta_plus, delta_minus = _allocate_span(parameter, requested_span)
        actual_span = delta_plus + delta_minus
        signature = (delta_plus, delta_minus)
        if actual_span <= 0.0:
            stop_reason = "hardware_limited"
            break
        if signature == last_signature:
            stop_reason = "hardware_limited"
            break
        last_signature = signature

        results = {}
        for direction, delta in (("plus", delta_plus), ("minus", delta_minus)):
            if delta <= 0.0:
                continue
            cache_key = (direction, delta)
            result = cache.get(cache_key)
            if result is None:
                sign = 1.0 if direction == "plus" else -1.0
                run_count, result = _run_single(
                    simulator,
                    parameter,
                    defaults,
                    value=float(parameter.default) + sign * delta,
                    label=f"{direction}={_format_delta(delta)}",
                    delta=delta,
                    seed=seed,
                    run_count=run_count,
                    total_runs=total_runs,
                    verbose=verbose,
                )
                cache[cache_key] = result
            results[direction] = result

        plus = results.get("plus")
        minus = results.get("minus")
        if plus is not None and plus.success and minus is not None and minus.success:
            score_diff = float(plus.score_val) - float(minus.score_val)
            scheme = "central"
            measured_span = actual_span
        elif plus is not None and plus.success:
            score_diff = float(plus.score_val) - float(baseline.score_val)
            scheme = "forward"
            measured_span = delta_plus
        elif minus is not None and minus.success:
            score_diff = float(minus.score_val) - float(baseline.score_val)
            scheme = "backward"
            measured_span = delta_minus
        else:
            # A completely failed interval is an unsafe upper bound.  Search
            # below it; if a successful lower point exists this also creates
            # the bracket used by the next iteration.
            upper_span = (
                requested_span
                if upper_span is None
                else min(upper_span, requested_span)
            )
            requested_span = _next_span(
                current_span=requested_span,
                score_magnitude=None,
                target=target_score_diff,
                lower_span=lower_span,
                upper_span=upper_span if lower_span is not None else None,
            )
            if verbose:
                print("    both directions failed; reducing span", flush=True)
            continue

        magnitude = abs(score_diff)
        if not math.isfinite(magnitude):
            upper_span = measured_span if upper_span is None else min(upper_span, measured_span)
            requested_span = measured_span / 2.0
            continue

        relative_error = abs(magnitude - target_score_diff) / target_score_diff
        trial = {
            "tested_span": measured_span,
            "delta_plus": delta_plus if scheme != "backward" else None,
            "delta_minus": delta_minus if scheme != "forward" else None,
            "score_diff": score_diff,
            "absolute_score_diff": magnitude,
            "relative_target_error": relative_error,
            "scheme": scheme,
            "iteration": iteration,
        }
        if magnitude > 0.0 and (
            best is None or relative_error < best["relative_target_error"]
        ):
            best = trial

        if verbose:
            print(
                f"    {scheme}: span={_format_delta(measured_span)} "
                f"|delta score|={magnitude:.6g} error={relative_error:.1%}",
                flush=True,
            )

        if magnitude > 0.0 and relative_error <= tolerance:
            stop_reason = "converged"
            break

        if magnitude < target_score_diff:
            lower_span = measured_span if lower_span is None else max(lower_span, measured_span)
        else:
            upper_span = measured_span if upper_span is None else min(upper_span, measured_span)

        requested_span = _next_span(
            current_span=measured_span,
            score_magnitude=magnitude,
            target=target_score_diff,
            lower_span=lower_span,
            upper_span=upper_span,
        )

    if best is None:
        status = "no_signal" if stop_reason != "hardware_limited" else stop_reason
        return run_count, _fallback_record(parameter, seed, status, iterations_done)

    refined = target_score_diff * best["tested_span"] / best["absolute_score_diff"]
    converged = best["relative_target_error"] <= tolerance
    status = "ok" if converged else "best_effort"
    return run_count, {
        "sensitivity": refined,
        "initial_sensitivity": initial_span,
        **{key: value for key, value in best.items() if key != "iteration"},
        "iterations": iterations_done,
        "best_iteration": best["iteration"],
        "seed": seed,
        "measured": True,
        "converged": converged,
        "status": status,
        "stop_reason": stop_reason,
    }


def compute_refined_sensitivity(
    simulator,
    *,
    parameters: Iterable = PARAMETERS,
    seed: Optional[int] = None,
    target_score_diff: float = DEFAULT_TARGET_SCORE_DIFF,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = 8,
    max_baseline_attempts: int = DEFAULT_MAX_BASELINE_ATTEMPTS,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Refine sensitivities using one independent common-random seed per parameter.

    If a parameter's baseline run fails (e.g. an unlucky Monte Carlo draw
    loses the whole beam even at nominal parameter values), a fresh seed is
    drawn and retried up to ``max_baseline_attempts`` times before the
    parameter is reported as ``baseline_failed``.
    """
    parameters = tuple(parameters)
    if target_score_diff <= 0.0:
        raise ValueError("target_score_diff must be > 0")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be between 0 and 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if max_baseline_attempts < 1:
        raise ValueError("max_baseline_attempts must be >= 1")

    rng = np.random.default_rng(seed)
    used_seeds: set = set()
    defaults = default_params()
    total_runs = len(parameters) * (max_baseline_attempts + 2 * max_iterations)
    run_count = 0
    records: Dict[str, dict] = {}
    for parameter in parameters:
        run_count, record = _refine_parameter(
            simulator,
            parameter,
            defaults,
            rng=rng,
            used_seeds=used_seeds,
            target_score_diff=target_score_diff,
            tolerance=tolerance,
            max_iterations=max_iterations,
            max_baseline_attempts=max_baseline_attempts,
            run_count=run_count,
            total_runs=total_runs,
            verbose=verbose,
        )
        records[parameter.name] = record
    return records


def print_report(records: Dict[str, dict]) -> None:
    header = (
        f"{'Parameter':<14} {'old':>12} {'refined':>12} {'span':>12} "
        f"{'|dscore|':>10} {'error':>9} {'scheme':>10} {'status':>12}"
    )
    separator = "=" * len(header)
    print(f"\n{separator}\n{header}\n{separator}")
    for parameter in PARAMETERS:
        record = records[parameter.name]
        span = record.get("tested_span")
        score_diff = record.get("absolute_score_diff")
        error = record.get("relative_target_error")
        print(
            f"{parameter.name:<14} {parameter.sensitivity:>12.4e} "
            f"{record['sensitivity']:>12.4e} "
            f"{('-' if span is None else _format_delta(span)):>12} "
            f"{('-' if score_diff is None else f'{score_diff:.4f}'):>10} "
            f"{('-' if error is None else f'{error:.1%}'):>9} "
            f"{(record.get('scheme') or '-'):>10} {record['status']:>12}"
        )
    print(separator)
    print("\n# Proposed sensitivity values (copy into ParameterSpec entries):")
    for parameter in PARAMETERS:
        print(f'"{parameter.name}": {records[parameter.name]["sensitivity"]:.17g},')


def save_report(
    records: Dict[str, dict],
    output: str | Path,
    *,
    target_score_diff: float,
    tolerance: float,
    run_config: dict,
    parameters: Iterable = PARAMETERS,
) -> Path:
    by_name = {parameter.name: parameter for parameter in parameters}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "iterative_current_sensitivity_refinement",
        "target_score_diff": target_score_diff,
        "tolerance": tolerance,
        "run_config": run_config,
        "new_sensitivity": {
            name: record["sensitivity"] for name, record in records.items()
        },
        "parameters": {
            name: {
                "key": by_name[name].key,
                "default": by_name[name].default,
                "old_sensitivity": by_name[name].sensitivity,
                **record,
            }
            for name, record in records.items()
        },
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    from beam_optimization.config.paths import (
        DEFAULT_REFINING_SENSITIVITY_OUTPUT, resolve_tracewin_project,
    )
    from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import TraceWinSimulator

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workspace", default=None, metavar="PATH")
    source.add_argument("--tracewin", default=None, metavar="INI")
    parser.add_argument("--calc-dir", default=None, metavar="PATH")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_REFINING_SENSITIVITY_OUTPUT),
        metavar="JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master RNG seed for a reproducible sequence of per-parameter seeds.",
    )
    parser.add_argument("--target-score-diff", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument(
        "--max-baseline-attempts",
        type=int,
        default=DEFAULT_MAX_BASELINE_ATTEMPTS,
        help=(
            "Retry a parameter's baseline run with a fresh seed up to this "
            "many times before giving up (default: %(default)s)."
        ),
    )
    parser.add_argument("--tracewin-threads", type=int, default=None, metavar="N")
    parser.add_argument("--tracewin-particles", type=int, default=10000)
    parser.add_argument("--tracewin-particle-key", default="nbr_part1")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    try:
        workspace, project_file = resolve_tracewin_project(
            workspace=args.workspace,
            tracewin=args.tracewin,
        )
    except ValueError as exc:
        parser.error(str(exc))

    calc_dir = (
        Path(args.calc_dir).expanduser().resolve()
        if args.calc_dir
        else workspace / "refining_sensitivity_calc"
    )
    output = Path(args.output).expanduser().resolve()
    print(f"TraceWin workspace : {workspace}")
    print(f"TraceWin project   : {project_file}")
    print(f"Calc dir           : {calc_dir}")
    print(f"JSON output        : {output}")
    print(f"Master seed        : {args.seed if args.seed is not None else 'random'}")
    print(f"Target score diff  : {args.target_score_diff}")
    print(f"Tolerance          : {args.tolerance:.1%}")
    print(f"Max iterations     : {args.max_iterations}")
    print(f"Max baseline retry : {args.max_baseline_attempts}")

    simulator = TraceWinSimulator(
        project_file=str(project_file),
        calc_dir=str(calc_dir),
        timeout=args.timeout,
        retries=args.retries,
        tracewin_params={args.tracewin_particle_key: int(args.tracewin_particles)},
        num_threads=args.tracewin_threads,
        initial_npart=args.tracewin_particles,
    )
    try:
        records = compute_refined_sensitivity(
            simulator,
            seed=args.seed,
            target_score_diff=args.target_score_diff,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            max_baseline_attempts=args.max_baseline_attempts,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print_report(records)
    saved = save_report(
        records,
        output,
        target_score_diff=args.target_score_diff,
        tolerance=args.tolerance,
        run_config={
            "workspace": str(workspace),
            "project": str(project_file),
            "calc_dir": str(calc_dir),
            "master_seed": args.seed,
            "seed_origin": "provided" if args.seed is not None else "system_entropy",
            "target_score_diff": args.target_score_diff,
            "tolerance": args.tolerance,
            "max_iterations": args.max_iterations,
            "max_baseline_attempts": args.max_baseline_attempts,
            "tracewin_threads": args.tracewin_threads,
            "tracewin_particles": args.tracewin_particles,
            "tracewin_particle_key": args.tracewin_particle_key,
            "timeout": args.timeout,
            "retries": args.retries,
        },
    )
    print(f"JSON saved to: {saved}")


if __name__ == "__main__":
    main()
