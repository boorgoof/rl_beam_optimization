"""Cold-start Bayesian Optimization directly against TraceWin.

No dataset or neural surrogate is loaded. The first evaluations follow a
Sobol low-discrepancy sequence; subsequent points are selected by a Gaussian
Process using Expected Improvement.
"""
from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from beam_optimization.algorithms.baselines.bayesian_opt import (
    hardware_aware_bounds,
)
from beam_optimization.config.adige import BAYESIAN_SCALE, PARAMETERS
from beam_optimization.config.paths import (
    DEFAULT_BAYESIAN_RESULTS_DIR,
    DEFAULT_TRACEWIN_INI,
    resolve_tracewin_project,
)
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import (
    TraceWinSimulator,
)
from beam_optimization.scripts.bayesian_opt import (
    _default_calc_root,
    _ensure_default_evaluation,
    _load_or_create_report,
    _print_best_params,
    _write_json_atomic,
    run_tracewin_bayesian,
    save_convergence_plot,
    save_delta_plot,
)


DEFAULT_OUTPUT = str(
    DEFAULT_BAYESIAN_RESULTS_DIR / "bayesian_opt_cold_start.json"
)
DEFAULT_SAMPLES_OUTPUT = str(
    DEFAULT_BAYESIAN_RESULTS_DIR / "bayesian_opt_cold_start_samples.pt"
)
REPORT_MODE = "tracewin_cold_start"


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _resolve_optimizer_seed(requested: int | None, output: Path) -> int:
    if requested is not None:
        return int(requested)
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("mode") == REPORT_MODE:
            saved = payload.get("config", {}).get("seed")
            if saved is not None:
                return int(saved)
    return secrets.randbelow(2**31)


def _resolve_tracewin_seed_base(
    requested: int | None,
    output: Path,
) -> int | None:
    if requested is not None:
        return int(requested)
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("mode") == REPORT_MODE:
            saved_config = payload.get("config", {})
            if "tracewin_seed_base" in saved_config:
                saved = saved_config["tracewin_seed_base"]
                return None if saved is None else int(saved)
    return None


def _report_config(
    args,
    *,
    project_file: Path,
    calc_root: Path,
    bounds: list[tuple[float, float]],
) -> dict:
    return {
        "project": str(project_file),
        "calc_root": str(calc_root),
        "initial_points": args.initial_points,
        "guided_calls": args.guided_calls,
        "total_calls": args.initial_points + args.guided_calls,
        "initial_point_generator": "sobol",
        "bounds_scale": args.bounds_scale,
        "bounds": {
            parameter.name: [float(lower), float(upper)]
            for parameter, (lower, upper) in zip(PARAMETERS, bounds)
        },
        "seed": args.seed,
        "tracewin_seed_base": args.tracewin_seed_base,
        "tracewin_particles": args.tracewin_particles,
        "tracewin_threads": args.tracewin_threads,
        "timeout": args.timeout,
        "retries": args.retries,
        "samples_output": str(Path(args.samples_output).expanduser().resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    tracewin_source = parser.add_mutually_exclusive_group()
    tracewin_source.add_argument("--workspace", default=None, metavar="PATH")
    tracewin_source.add_argument(
        "--tracewin",
        default=None,
        metavar="INI",
        help=f"TraceWin project file (default: {DEFAULT_TRACEWIN_INI})",
    )
    parser.add_argument("--calc-dir", default=None, metavar="PATH")
    parser.add_argument("--initial-points", type=int, default=64)
    parser.add_argument("--guided-calls", type=int, default=100)
    parser.add_argument(
        "--bounds-scale",
        type=float,
        default=BAYESIAN_SCALE,
        help=(
            "Search half-width in sensitivity units around each default "
            "(default: BAYESIAN_SCALE from adige.py)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sobol/GP seed (default: random once, then stored in checkpoint)",
    )
    parser.add_argument(
        "--tracewin-seed-base",
        type=int,
        default=None,
        help=(
            "Deterministic first TraceWin seed. If omitted, no random_seed "
            "parameter is passed to TraceWin."
        ),
    )
    parser.add_argument("--tracewin-particles", type=int, default=10_000)
    parser.add_argument("--tracewin-threads", type=int, default=None, metavar="N")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-output", default=DEFAULT_SAMPLES_OUTPUT)
    args = parser.parse_args()

    if not _is_power_of_two(args.initial_points):
        parser.error("--initial-points must be a positive power of two for Sobol")
    if args.guided_calls < 0:
        parser.error("--guided-calls must be non-negative")

    try:
        workspace, project_file = resolve_tracewin_project(
            workspace=args.workspace,
            tracewin=args.tracewin,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output = Path(args.output).expanduser().resolve()
    args.seed = _resolve_optimizer_seed(args.seed, output)
    args.tracewin_seed_base = _resolve_tracewin_seed_base(
        args.tracewin_seed_base,
        output,
    )
    bounds = hardware_aware_bounds(PARAMETERS, args.bounds_scale)
    samples_output = Path(args.samples_output).expanduser().resolve()
    calc_root = (
        Path(args.calc_dir).expanduser().resolve()
        if args.calc_dir
        else _default_calc_root(workspace, output)
    )
    config = _report_config(
        args,
        project_file=project_file,
        calc_root=calc_root,
        bounds=bounds,
    )
    try:
        report = _load_or_create_report(
            output,
            config=config,
            warm_start=[],
            mode=REPORT_MODE,
        )
    except ValueError as exc:
        parser.error(str(exc))

    total_calls = args.initial_points + args.guided_calls
    print(f"TraceWin workspace : {workspace}")
    print(f"TraceWin project   : {project_file}")
    print("Dataset            : none (cold start)")
    print(f"Sobol points       : {args.initial_points}")
    print(f"GP-guided calls    : {args.guided_calls}")
    print(f"Total calls        : {total_calls}")
    print(f"Bounds scale       : {args.bounds_scale} sensitivity units")
    print(f"Sobol/GP seed      : {args.seed}")
    print(
        "TraceWin seeds     : "
        + (
            "unset (TraceWin default behavior)"
            if args.tracewin_seed_base is None
            else f"{args.tracewin_seed_base} + evaluation index"
        )
    )
    print(f"Checkpoint         : {output}")

    _ensure_default_evaluation(
        report,
        project_file=project_file,
        calc_root=calc_root,
        timeout=args.timeout,
        retries=args.retries,
        tracewin_particles=args.tracewin_particles,
        tracewin_threads=args.tracewin_threads,
    )
    _write_json_atomic(output, report)

    def simulator_factory(run_index: int) -> TraceWinSimulator:
        return TraceWinSimulator(
            project_file=str(project_file),
            calc_dir=str(calc_root / f"run_{run_index:03d}"),
            timeout=args.timeout,
            retries=args.retries,
            tracewin_params={"nbr_part1": args.tracewin_particles},
            num_threads=args.tracewin_threads,
            initial_npart=args.tracewin_particles,
        )

    report = run_tracewin_bayesian(
        simulator_factory=simulator_factory,
        source_dataset=None,
        bounds=bounds,
        report=report,
        output=output,
        new_samples_output=samples_output,
        merged_dataset_output=None,
        n_calls=total_calls,
        n_runs=1,
        seed=args.seed,
        tracewin_seed_base=args.tracewin_seed_base,
        initial_points=args.initial_points,
        initial_point_generator="sobol",
    )

    convergence = save_convergence_plot(
        report,
        output,
        prefix="bayesian_opt_cold_start",
    )
    best_result = report["best_result"]
    delta = None
    if best_result is not None:
        _print_best_params(best_result, report.get("default_result"))
        delta = save_delta_plot(
            best_result,
            output,
            prefix="bayesian_opt_cold_start",
            default_result=report.get("default_result"),
        )
    else:
        print("\nWARNING: every TraceWin evaluation failed; no best point exists.")

    print(f"\nJSON checkpoint      -> {output}")
    print(f"Valid TraceWin data  -> {samples_output}")
    if convergence is not None:
        print(f"Convergence plot     -> {convergence}")
    if delta is not None:
        print(f"Delta plot           -> {delta}")


if __name__ == "__main__":
    main()
