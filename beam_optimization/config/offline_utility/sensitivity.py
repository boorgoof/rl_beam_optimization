"""Offline TraceWin sensitivity analysis for ADIGE parameters.

For each parameter, the search evaluates a common-random-number pair
``score(p + delta) - score(p - delta)`` and increases ``delta`` until the
requested score difference is reached or both directions are pinned at their
hardware limit. Each parameter draws its own independent TraceWin seed from
an RNG (seeded by ``seed`` if given, otherwise from system entropy): the seed
is shared across one parameter's own baseline/+delta/-delta probes (so the
Monte Carlo noise cancels out of that parameter's difference), but is not
reused across different parameters, so one unlucky draw cannot correlate the
measurements of all 18 parameters at once. Passing the same ``seed`` again
reproduces the exact same sequence of per-parameter seeds; the seed actually
used for each parameter is also recorded in its own result record.

When a requested delta would push a parameter past its hardware bound
(``hw_min``/``hw_max``) on one or both sides, that side is not dropped: it is
clipped to the maximum displacement the hardware allows, so a central
difference stays available (at reduced span) even right next to a bound.
Once a side is pinned at its hardware limit it is never re-simulated (the
result is cached and reused), since escalating the requested delta further
cannot change an already-capped side's outcome.

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
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

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


def _direction_limit(parameter, direction: str) -> float:
    """Maximum |delta| the hardware bound allows for one direction (inf if unbounded)."""
    if direction == "plus":
        upper = parameter.hw_max
        if upper is None:
            return math.inf
        return max(0.0, float(upper) - float(parameter.default))
    lower = parameter.hw_min
    if lower is None and parameter.name.startswith("AD.SO"):
        lower = 0.0  # solenoids never go negative without an explicit hw_min
    if lower is None:
        return math.inf
    return max(0.0, float(parameter.default) - float(lower))


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


def compute_sensitivity(
    simulator,
    *,
    seed: Optional[int] = None,
    escalation_factor: float = 3.0,
    max_iterations: int = 8,
    target_score_diff: float = 1.0,
    initial_delta_provider: Optional[Callable[[object], float]] = None,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Compute one sensitivity per parameter, each with its own TraceWin seed.

    ``seed`` seeds the RNG that draws one independent per-parameter seed
    (shared only across that parameter's own baseline/+delta/-delta probes);
    omit it to draw a fresh, non-reproducible sequence from system entropy.
    """
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
    rng = np.random.default_rng(seed)

    for parameter in PARAMETERS:
        initial_delta = float(
            initial_delta_provider(parameter)
            if initial_delta_provider is not None
            else _initial_delta(parameter)
        )
        if not math.isfinite(initial_delta) or initial_delta <= 0.0:
            raise ValueError(
                f"invalid initial delta for {parameter.name}: {initial_delta!r}"
            )
        param_seed = int(rng.integers(0, 2**31))
        run_count, record = _search_parameter(
            simulator,
            parameter,
            defaults,
            initial_delta=initial_delta,
            seed=param_seed,
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
    initial_delta: float,
    seed: int,
    escalation_factor: float,
    max_iterations: int,
    target_score_diff: float,
    run_count: int,
    total_runs: int,
    verbose: bool,
) -> tuple[int, dict]:
    delta = initial_delta
    last_good_delta: Optional[float] = None
    best_span: Optional[float] = None
    best_diff: Optional[float] = None
    best_scheme: Optional[str] = None
    best_iteration: Optional[int] = None
    best_delta_plus: Optional[float] = None
    best_delta_minus: Optional[float] = None
    hardware_limited = False

    if verbose:
        print(
            f"\n{parameter.name}: seed={seed} starting delta = {_format_delta(delta)}",
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
            "delta_plus": None,
            "delta_minus": None,
            "diff": None,
            "scheme": None,
            "iterations": 0,
            "status": "baseline_failed",
            "seed": seed,
        }

    # Per-direction hardware ceiling on |delta|, and the last effective delta
    # (and its result) actually probed on each side -- once a side is pinned
    # at its ceiling, escalating the requested delta further cannot change
    # its outcome, so it is reused instead of re-simulated.
    limits = {
        "plus": _direction_limit(parameter, "plus"),
        "minus": _direction_limit(parameter, "minus"),
    }
    last_tested: Dict[str, Optional[float]] = {"plus": None, "minus": None}
    last_result: Dict[str, object] = {"plus": None, "minus": None}

    for iteration in range(1, max_iterations + 1):
        tested_this_iteration = False
        probed: Dict[str, tuple[float, object]] = {}

        for direction in ("plus", "minus"):
            effective_delta = min(delta, limits[direction])
            if effective_delta <= 0.0:
                continue  # default already at/past the hardware bound on this side

            previous = last_tested[direction]
            if previous is not None and math.isclose(effective_delta, previous):
                probed[direction] = (effective_delta, last_result[direction])
                continue

            sign = 1.0 if direction == "plus" else -1.0
            value = float(parameter.default) + sign * effective_delta
            label = "+delta" if direction == "plus" else "-delta"
            run_count, result = _run_single(
                simulator,
                parameter,
                defaults,
                value=value,
                label=f"{label}={_format_delta(effective_delta)}",
                delta=effective_delta,
                seed=seed,
                run_count=run_count,
                total_runs=total_runs,
                verbose=verbose,
            )
            last_tested[direction] = effective_delta
            last_result[direction] = result
            probed[direction] = (effective_delta, result)
            tested_this_iteration = True

        plus = probed.get("plus")
        minus = probed.get("minus")
        delta_plus = plus[0] if plus is not None else None
        delta_minus = minus[0] if minus is not None else None
        diff = None
        scheme = None
        span = None

        if plus is not None and plus[1].success and minus is not None and minus[1].success:
            diff = float(plus[1].score_val) - float(minus[1].score_val)
            span = delta_plus + delta_minus
            scheme = "central"
        elif plus is not None and plus[1].success:
            diff = float(plus[1].score_val) - float(baseline.score_val)
            span = delta_plus
            scheme = "forward"
        elif minus is not None and minus[1].success:
            diff = float(minus[1].score_val) - float(baseline.score_val)
            span = delta_minus
            scheme = "backward"

        if diff is None:
            if not tested_this_iteration:
                # Both sides are either permanently unusable or already
                # pinned at their hardware ceiling with no usable result:
                # nothing will change by escalating delta further.
                hardware_limited = True
                if verbose:
                    print(
                        f"    delta={_format_delta(delta)}: no usable direction "
                        "left to try; stopping",
                        flush=True,
                    )
                break
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
            best_span = span
            best_diff = diff
            best_scheme = scheme
            best_iteration = iteration
            best_delta_plus = delta_plus
            best_delta_minus = delta_minus

        if abs(diff) >= target_score_diff:
            sensitivity = TARGET_SCORE_CHANGE * span / abs(diff)
            if verbose:
                print(
                    f"    ACCEPTED {scheme}: span={_format_delta(span)} diff={diff:.4f} "
                    f"sensitivity={sensitivity:.6e}",
                    flush=True,
                )
            return run_count, {
                "sensitivity": sensitivity,
                "delta": span,
                "delta_plus": delta_plus,
                "delta_minus": delta_minus,
                "diff": diff,
                "scheme": scheme,
                "iterations": iteration,
                "status": "ok",
                "seed": seed,
            }

        if verbose:
            print(
                f"    {scheme}: diff={diff:.4f} below target "
                f"{target_score_diff:g}; escalating",
                flush=True,
            )
        last_good_delta = delta

        if not tested_this_iteration:
            # Both sides already pinned at their hardware ceiling and still
            # below target: escalating the requested delta changes nothing.
            hardware_limited = True
            break

        delta *= escalation_factor

    if best_diff is None or best_span is None:
        # No perturbation ever produced a usable finite difference, even
        # after backing off/escalating delta from the initial guess. Rather
        # than silently keeping the old (possibly wrong -- and possibly the
        # reason the search never got a signal) declared sensitivity,
        # estimate a conservative value from the initial delta that was
        # actually tried first: treat it as if it alone would span
        # target_score_diff points, which gives a small/safe sensitivity.
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
            "delta_plus": None,
            "delta_minus": None,
            "diff": None,
            "scheme": None,
            "iterations": max_iterations,
            "status": "no_diff_estimated_from_initial_delta",
            "seed": seed,
        }

    effective_diff = max(abs(best_diff), target_score_diff)
    sensitivity = TARGET_SCORE_CHANGE * best_span / effective_diff
    status = "hardware_limited" if hardware_limited else "weak_signal"
    if verbose:
        print(
            f"    SELECTED {best_scheme}: span={_format_delta(best_span)} "
            f"sensitivity={sensitivity:.6e} status={status}",
            flush=True,
        )
    return run_count, {
        "sensitivity": sensitivity,
        "delta": best_span,
        "delta_plus": best_delta_plus,
        "delta_minus": best_delta_minus,
        "diff": best_diff,
        "scheme": best_scheme,
        "iterations": best_iteration,
        "status": status,
        "seed": seed,
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
        "algorithm": "per_parameter_seed_finite_difference_with_hardware_clipping",
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
            "ADIGE sensitivity from TraceWin finite differences, one "
            "independent TraceWin seed drawn per parameter. Deltas exceeding "
            "a hardware bound are clipped to the maximum allowed displacement "
            "instead of being dropped. Prints and saves one value per "
            "parameter without modifying adige.py."
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "RNG seed used to draw one independent TraceWin seed per "
            "parameter. Given: the per-parameter sequence is reproducible "
            "across reruns. Omitted (default): a fresh independent seed is "
            "drawn for every parameter from system entropy; the seed "
            "actually used for each parameter is still recorded in its "
            "result record."
        ),
    )
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
    print(
        f"Seed                  : {args.seed} "
        f"({'reproducible per-parameter sequence' if args.seed is not None else 'random per parameter'})"
    )
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
            "project": str(project_file),
            "calc_dir": str(calc_dir.expanduser().resolve()),
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
