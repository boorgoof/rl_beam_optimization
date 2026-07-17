"""Offline calculation of operational TraceWin parameter bounds.

All ADIGE parameters except the AD.EM family are checked.  During each check,
all other parameters remain at their defaults.  The results are operational
transport limits, not certified hardware limits.

Example:
    python -m beam_optimization parameter_bounds_calculation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from beam_optimization.config.adige import PARAMETERS
from beam_optimization.config.paths import (
    DEFAULT_PARAMETER_BOUNDS_OUTPUT,
    DEFAULT_TRACEWIN_INI,
    resolve_tracewin_project,
)
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import (
    TraceWinSimulator,
)


TARGET_NAMES = tuple(
    parameter.name for parameter in PARAMETERS if not parameter.name.startswith("AD.EM")
)
BAD_VALUE_MARKERS = (
    "bad value",
    "beam distribution never reaches the end of the field map",
    "synchronous particle never reaches the end of the field map",
)
PARTICLE_LOSS_MARKERS = (
    "particle lost",
    "particles lost",
    "lost particle",
    "lost particles",
    "all particles",
    "transport failed",
    "losses in field_map",
)


def _tracewin_accepts_value(result) -> bool:
    """Classify a run solely for parameter-bound verification.

    Generic beam/particle loss is accepted because it does not mean that
    TraceWin rejected the supplied element value.  Explicit ``bad value`` and
    failure to reach the end of a field map mark a bound failure.  Every other
    failure, including a timeout, also marks a failed bound probe instead of
    aborting the complete scan.
    """
    if result.success:
        return True

    error = str(result.error or "")
    normalized_error = error.casefold()
    if any(marker in normalized_error for marker in BAD_VALUE_MARKERS):
        print("      TraceWin rejected the parameter or field-map transport failed")
        return False

    if any(marker in normalized_error for marker in PARTICLE_LOSS_MARKERS):
        print("      particle loss ignored for bound verification")
        return True

    print("      TraceWin run failed; treating this probe as a bound failure")
    return False


def _refine_boundary(
    is_success: Callable[[float], bool],
    *,
    success_value: float,
    failure_value: float,
    tolerance: float,
    max_bisections: int,
) -> tuple[float, float]:
    """Bisect a known success/failure bracket."""
    for _ in range(max_bisections):
        if abs(failure_value - success_value) <= tolerance:
            break
        midpoint = 0.5 * (success_value + failure_value)
        print(f"      refining {midpoint:.10g}")
        if is_success(midpoint):
            success_value = midpoint
        else:
            failure_value = midpoint
    return success_value, failure_value


def _expand_from_success(
    is_success: Callable[[float], bool],
    *,
    last_success: float,
    direction: int,
    initial_step: float,
    growth_factor: float,
    tolerance: float,
    max_expansions: int,
    max_bisections: int,
) -> dict:
    """Expand geometrically until failure, then refine the boundary."""
    step = float(initial_step)
    for expansion in range(1, max_expansions + 1):
        candidate = last_success + direction * step
        print(
            f"      expansion {expansion}/{max_expansions}: "
            f"testing {candidate:.10g} (step={direction * step:+.4g})"
        )
        if not is_success(candidate):
            refined_success, refined_failure = _refine_boundary(
                is_success,
                success_value=last_success,
                failure_value=candidate,
                tolerance=tolerance,
                max_bisections=max_bisections,
            )
            return {
                "last_success": refined_success,
                "first_failure": refined_failure,
                "expansions": expansion,
                "boundary_found": True,
            }
        last_success = candidate
        step *= growth_factor

    return {
        "last_success": last_success,
        "first_failure": None,
        "expansions": max_expansions,
        "boundary_found": False,
    }


def _search_inward_from_failed_bound(
    is_success: Callable[[float], bool],
    *,
    failed_bound: float,
    direction: int,
    initial_step: float,
    growth_factor: float,
    tolerance: float,
    max_expansions: int,
    max_bisections: int,
) -> dict:
    """Search inward when the default cannot bracket a failed declared bound."""
    step = float(initial_step)
    for expansion in range(1, max_expansions + 1):
        candidate = failed_bound - direction * step
        print(
            f"      inward search {expansion}/{max_expansions}: "
            f"testing {candidate:.10g}"
        )
        if is_success(candidate):
            last_success, first_failure = _refine_boundary(
                is_success,
                success_value=candidate,
                failure_value=failed_bound,
                tolerance=tolerance,
                max_bisections=max_bisections,
            )
            return {
                "last_success": last_success,
                "first_failure": first_failure,
                "expansions": expansion,
                "boundary_found": True,
            }
        step *= growth_factor

    return {
        "last_success": None,
        "first_failure": failed_bound,
        "expansions": max_expansions,
        "boundary_found": False,
    }


def _verify_direction(
    is_success: Callable[[float], bool],
    *,
    default: float,
    direction: int,
    declared_bound: float | None,
    initial_step: float,
    outside_step_factor: float,
    growth_factor: float,
    tolerance: float,
    max_expansions: int,
    max_bisections: int,
) -> dict:
    """Verify one declared bound, expanding or contracting when needed."""
    side = "lower" if direction < 0 else "upper"

    if declared_bound is None:
        print("      no declared bound; starting from the default")
        search = _expand_from_success(
            is_success,
            last_success=default,
            direction=direction,
            initial_step=initial_step,
            growth_factor=growth_factor,
            tolerance=tolerance,
            max_expansions=max_expansions,
            max_bisections=max_bisections,
        )
        status = (
            "missing_declared_bound_estimated"
            if search["boundary_found"]
            else "missing_declared_bound_not_found_after_max_expansions"
        )
        return {
            "side": side,
            "declared_bound": None,
            "at_bound_success": None,
            "outside_probe": None,
            "status": status,
            "suggested_bound": search["last_success"],
            **search,
        }

    declared_bound = float(declared_bound)
    print(f"      testing declared bound {declared_bound:.10g}")
    at_bound_success = bool(is_success(declared_bound))
    if not at_bound_success:
        default_is_inward = direction * (declared_bound - default) >= 0
        if default_is_inward:
            last_success, first_failure = _refine_boundary(
                is_success,
                success_value=default,
                failure_value=declared_bound,
                tolerance=tolerance,
                max_bisections=max_bisections,
            )
            adjustment = {
                "last_success": last_success,
                "first_failure": first_failure,
                "expansions": 0,
                "boundary_found": True,
            }
        else:
            print("      default is outside the declared interval; searching inward")
            adjustment = _search_inward_from_failed_bound(
                is_success,
                failed_bound=declared_bound,
                direction=direction,
                initial_step=initial_step,
                growth_factor=growth_factor,
                tolerance=tolerance,
                max_expansions=max_expansions,
                max_bisections=max_bisections,
            )
        status = (
            "declared_bound_contracted"
            if adjustment["boundary_found"]
            else "failed_bound_no_inward_success_found"
        )
        return {
            "side": side,
            "declared_bound": declared_bound,
            "at_bound_success": False,
            "outside_probe": None,
            "status": status,
            "suggested_bound": adjustment["last_success"],
            **adjustment,
        }

    outside_step = max(
        initial_step * outside_step_factor,
        abs(declared_bound - default) * 1e-6,
        1e-12,
    )
    outside_value = declared_bound + direction * outside_step
    print(f"      testing just outside {outside_value:.10g}")
    outside_success = bool(is_success(outside_value))
    outside_probe = {"value": outside_value, "success": outside_success}

    if not outside_success:
        return {
            "side": side,
            "declared_bound": declared_bound,
            "at_bound_success": True,
            "outside_probe": outside_probe,
            "status": "declared_bound_confirmed",
            "suggested_bound": declared_bound,
            "last_success": declared_bound,
            "first_failure": outside_value,
            "expansions": 0,
            "boundary_found": True,
        }

    search = _expand_from_success(
        is_success,
        last_success=outside_value,
        direction=direction,
        initial_step=outside_step * growth_factor,
        growth_factor=growth_factor,
        tolerance=tolerance,
        max_expansions=max_expansions,
        max_bisections=max_bisections,
    )
    status = (
        "declared_bound_expanded"
        if search["boundary_found"]
        else "expanded_but_not_found_after_max_expansions"
    )
    return {
        "side": side,
        "declared_bound": declared_bound,
        "at_bound_success": True,
        "outside_probe": outside_probe,
        "status": status,
        "suggested_bound": search["last_success"],
        **search,
    }


def verify_transport_limits(
    simulator,
    *,
    initial_step_factor: float = 5.0,
    outside_step_factor: float = 1.0,
    growth_factor: float = 2.0,
    tolerance_factor: float = 0.1,
    max_expansions: int = 16,
    max_bisections: int = 16,
) -> dict:
    """Verify and, when needed, adjust target transport limits."""
    if initial_step_factor <= 0 or outside_step_factor <= 0:
        raise ValueError("step factors must be positive")
    if tolerance_factor <= 0:
        raise ValueError("tolerance_factor must be positive")
    if growth_factor <= 1:
        raise ValueError("growth_factor must be greater than 1")
    if max_expansions <= 0 or max_bisections <= 0:
        raise ValueError("expansion and bisection counts must be positive")

    by_name = {parameter.name: parameter for parameter in PARAMETERS}
    missing = [name for name in TARGET_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"Missing parameters in adige.py: {missing}")

    print("Checking the all-default TraceWin configuration...")
    if not _tracewin_accepts_value(simulator.simulate()):
        raise RuntimeError(
            "TraceWin rejects the all-default configuration as a bad value; "
            "limits cannot be verified."
        )

    results = {}
    for name in TARGET_NAMES:
        parameter = by_name[name]
        initial_step = max(
            abs(parameter.sensitivity) * initial_step_factor,
            abs(parameter.default) * 1e-6,
            1e-12,
        )
        tolerance = max(abs(parameter.sensitivity) * tolerance_factor, 1e-12)

        def succeeds(value: float, *, key=parameter.key) -> bool:
            return _tracewin_accepts_value(
                simulator.simulate({key: float(value)})
            )

        print(f"\n{name}: default={parameter.default:.10g}")
        print("  verifying lower bound")
        lower = _verify_direction(
            succeeds,
            default=parameter.default,
            direction=-1,
            declared_bound=parameter.hw_min,
            initial_step=initial_step,
            outside_step_factor=outside_step_factor,
            growth_factor=growth_factor,
            tolerance=tolerance,
            max_expansions=max_expansions,
            max_bisections=max_bisections,
        )
        print("  verifying upper bound")
        upper = _verify_direction(
            succeeds,
            default=parameter.default,
            direction=1,
            declared_bound=parameter.hw_max,
            initial_step=initial_step,
            outside_step_factor=outside_step_factor,
            growth_factor=growth_factor,
            tolerance=tolerance,
            max_expansions=max_expansions,
            max_bisections=max_bisections,
        )
        results[name] = {
            "key": parameter.key,
            "default": parameter.default,
            "sensitivity": parameter.sensitivity,
            "declared_hw_min": parameter.hw_min,
            "declared_hw_max": parameter.hw_max,
            "default_within_declared_bounds": (
                (parameter.hw_min is None or parameter.default >= parameter.hw_min)
                and (parameter.hw_max is None or parameter.default <= parameter.hw_max)
            ),
            "tested_initial_step": initial_step,
            "lower": lower,
            "upper": upper,
            "suggested_transport_min": lower["suggested_bound"],
            "suggested_transport_max": upper["suggested_bound"],
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TraceWin transport bounds for all configured "
            "ADIGE parameters except the AD.EM family."
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
        help="TraceWin calculation directory (default: parameter_bounds_calc inside the workspace).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="TraceWin random_seed override. Default: unseeded (TraceWin picks its own).",
    )
    parser.add_argument("--tracewin-particles", type=int, default=10_000)
    parser.add_argument("--tracewin-threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--initial-step-factor", type=float, default=5.0)
    parser.add_argument("--outside-step-factor", type=float, default=1.0)
    parser.add_argument("--growth-factor", type=float, default=2.0)
    parser.add_argument("--tolerance-factor", type=float, default=0.1)
    parser.add_argument("--max-expansions", type=int, default=16)
    parser.add_argument("--max-bisections", type=int, default=16)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_PARAMETER_BOUNDS_OUTPUT),
        metavar="JSON",
        help="Result JSON path (default: %(default)s).",
    )
    args = parser.parse_args()

    try:
        workspace, project = resolve_tracewin_project(
            workspace=args.workspace,
            tracewin=args.tracewin,
        )
    except ValueError as exc:
        parser.error(str(exc))

    calc_dir = Path(args.calc_dir) if args.calc_dir else workspace / "parameter_bounds_calc"

    print(f"TraceWin workspace: {workspace}")
    print(f"TraceWin project:   {project}")
    print(f"Calc dir:           {calc_dir}")

    tracewin_params = {"nbr_part1": args.tracewin_particles}
    if args.seed is not None:
        tracewin_params["random_seed"] = args.seed

    simulator = TraceWinSimulator(
        project_file=str(project),
        calc_dir=str(calc_dir.expanduser().resolve()),
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=0.0,
        tracewin_params=tracewin_params,
        num_threads=args.tracewin_threads,
        initial_npart=args.tracewin_particles,
    )

    results = verify_transport_limits(
        simulator,
        initial_step_factor=args.initial_step_factor,
        outside_step_factor=args.outside_step_factor,
        growth_factor=args.growth_factor,
        tolerance_factor=args.tolerance_factor,
        max_expansions=args.max_expansions,
        max_bisections=args.max_bisections,
    )

    print("\nTraceWin parameter bounds:")
    for name, result in results.items():
        suggested_min = result["suggested_transport_min"]
        suggested_max = result["suggested_transport_max"]
        min_text = "not found" if suggested_min is None else f"{suggested_min:.10g}"
        max_text = "not found" if suggested_max is None else f"{suggested_max:.10g}"
        print(
            f"  {name}: min={min_text} ({result['lower']['status']}), "
            f"max={max_text} ({result['upper']['status']})"
        )

    report = {
        "workspace": str(workspace),
        "project": str(project),
        "target_names": list(TARGET_NAMES),
        "bound_failure_criterion": {
            "rejected": list(BAD_VALUE_MARKERS),
            "accepted_particle_loss_messages": list(PARTICLE_LOSS_MARKERS),
            "all_other_failures": "treat_as_bound_failure",
        },
        "scan_config": {
            "seed": args.seed,
            "tracewin_particles": args.tracewin_particles,
            "tracewin_threads": args.tracewin_threads,
            "timeout": args.timeout,
            "retries": args.retries,
            "initial_step_factor": args.initial_step_factor,
            "outside_step_factor": args.outside_step_factor,
            "growth_factor": args.growth_factor,
            "tolerance_factor": args.tolerance_factor,
            "max_expansions": args.max_expansions,
            "max_bisections": args.max_bisections,
        },
        "parameter_bounds": results,
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"JSON saved to: {output}")


if __name__ == "__main__":
    main()
