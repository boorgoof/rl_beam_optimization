"""Bayesian Optimization directly against TraceWin.

The Gaussian Process is warm-started with real TraceWin rows from a
BeamDataset. Every new point proposed by the optimizer is evaluated by
TraceWin. A deterministic per-evaluation seed sequence is used only when
``--tracewin-seed-base`` is explicitly provided.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from beam_optimization.algorithms.baselines.bayesian_opt import (
    hardware_aware_bounds,
    select_warm_start,
)
from beam_optimization.config.adige import (
    BAYESIAN_SCALE,
    ERROR_SCORE,
    PARAMETERS,
    PARAM_KEYS,
    default_params,
    sensitivity_vec,
)
from beam_optimization.config.paths import (
    DEFAULT_BAYESIAN_RESULTS_DIR,
    DEFAULT_TRACEWIN_INI,
    configure_matplotlib_cache,
    default_dataset_path,
    resolve_tracewin_project,
)
from beam_optimization.env.dataset import BeamDataset, tracewin_result_to_flat_sample
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import (
    TraceWinSimulator,
)


REPORT_VERSION = 1
DEFAULT_OUTPUT = str(DEFAULT_BAYESIAN_RESULTS_DIR / "bayesian_opt.json")
DEFAULT_NEW_SAMPLES_OUTPUT = str(
    DEFAULT_BAYESIAN_RESULTS_DIR / "bayesian_opt_tracewin_samples.pt"
)
DEFAULT_MERGED_DATASET_OUTPUT = str(
    DEFAULT_BAYESIAN_RESULTS_DIR / "dataset_with_bayesian.pt"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_calc_root(workspace: Path, output: Path) -> Path:
    digest = hashlib.sha256(str(output).encode("utf-8")).hexdigest()[:12]
    return Path(workspace) / f"tracewin_bayesian_{digest}"


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    temporary.replace(path)


class _GPFailurePenalty:
    """Replay-stable objective values for the GP.

    A failed simulation must not be told to the GP as ERROR_SCORE (-999):
    among real scores of order ~100 that is a huge outlier which wrecks the
    fitted length scales. Instead, failures are told a value slightly below
    the worst successful score seen so far. The mapping depends only on the
    ordered history of (score, success) pairs, so replaying the persisted
    evaluations on resume reproduces the exact same GP state. The full
    ERROR_SCORE is still recorded in the report for bookkeeping.
    """

    MARGIN = 10.0
    DEFAULT_FLOOR = 0.0

    def __init__(self, warm_start: list[dict]):
        scores = [float(entry["score"]) for entry in warm_start]
        self._worst_success: float | None = min(scores) if scores else None

    def tell_value(self, score: float, success: bool) -> float:
        if success:
            score = float(score)
            self._worst_success = (
                score
                if self._worst_success is None
                else min(self._worst_success, score)
            )
            return -score
        floor = (
            self._worst_success
            if self._worst_success is not None
            else self.DEFAULT_FLOOR
        )
        return -(floor - self.MARGIN)


def _params_from_vector(vector) -> dict[str, float]:
    return {key: float(value) for key, value in zip(PARAM_KEYS, vector)}


def _vector_from_params(params: dict[str, float]) -> list[float]:
    return [float(params[key]) for key in PARAM_KEYS]


def _warm_start_payload(selection) -> list[dict]:
    return [
        {
            "dataset_index": int(index),
            "selection": label,
            "params": _params_from_vector(vector),
            "score": float(score),
        }
        for index, label, vector, score in zip(
            selection.indices,
            selection.labels,
            selection.param_vectors,
            selection.scores,
        )
    ]


def _report_config(
    args,
    *,
    dataset_path: Path,
    project_file: Path,
    calc_root: Path,
    bounds: list[tuple[float, float]],
) -> dict:
    return {
        "dataset": str(dataset_path),
        "project": str(project_file),
        "calc_root": str(calc_root),
        "n_calls": args.n_calls,
        "n_runs": args.n_runs,
        "warm_best": args.warm_best,
        "warm_diverse": args.warm_diverse,
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
        "new_samples_output": str(Path(args.new_samples_output).resolve()),
        "merged_dataset_output": str(Path(args.merged_dataset_output).resolve()),
    }


def _new_report(
    config: dict,
    warm_start: list[dict],
    *,
    mode: str = "tracewin",
) -> dict:
    return {
        "version": REPORT_VERSION,
        "mode": mode,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "config": config,
        "warm_start": warm_start,
        "runs": [],
        "best_result": None,
    }


def _load_or_create_report(
    output: Path,
    *,
    config: dict,
    warm_start: list[dict],
    mode: str = "tracewin",
) -> dict:
    if not output.exists():
        return _new_report(config, warm_start, mode=mode)

    report = json.loads(output.read_text(encoding="utf-8"))
    if report.get("version") != REPORT_VERSION or report.get("mode") != mode:
        raise ValueError(
            f"{output} is not a compatible TraceWin Bayesian checkpoint. "
            "Move it or select a different --output path."
        )
    if report.get("config") != config:
        raise ValueError(
            f"{output} was created with a different configuration. "
            "Use matching arguments or a different --output path."
        )
    if report.get("warm_start") != warm_start:
        raise ValueError(
            f"{output} does not match the warm-start rows selected from the dataset."
        )
    report["status"] = "running"
    report["updated_at"] = _utc_now()
    return report


def _ensure_run(report: dict, run_index: int, optimizer_seed: int) -> dict:
    while len(report["runs"]) <= run_index:
        index = len(report["runs"])
        report["runs"].append(
            {
                "run_index": index,
                "optimizer_seed": int(optimizer_seed if index == run_index else 0),
                "status": "running",
                "evaluations": [],
                "best_score": None,
                "best_params": None,
            }
        )
    run = report["runs"][run_index]
    expected_seed = int(optimizer_seed)
    if run["optimizer_seed"] != expected_seed:
        raise ValueError(
            f"Run {run_index} checkpoint seed {run['optimizer_seed']} "
            f"does not match requested seed {expected_seed}"
        )
    run["status"] = "running"
    return run


def _completed_evaluation_count(report: dict) -> int:
    return sum(len(run.get("evaluations", [])) for run in report.get("runs", []))


def _random_unused_tracewin_seed(report: dict) -> int:
    used = {
        int(evaluation["tracewin_seed"])
        for run in report.get("runs", [])
        for evaluation in run.get("evaluations", [])
        if evaluation.get("tracewin_seed") is not None
    }
    while True:
        seed = secrets.randbelow(2**31)
        if seed not in used:
            return seed


def _load_new_samples(path: Path, report: dict) -> BeamDataset:
    expected = sum(
        1
        for run in report.get("runs", [])
        for evaluation in run.get("evaluations", [])
        if evaluation.get("success")
    )
    if path.exists():
        dataset = BeamDataset.load(path)
        if len(dataset) != expected:
            raise ValueError(
                f"{path} contains {len(dataset)} rows, but the Bayesian checkpoint "
                f"records {expected} successful TraceWin evaluations."
            )
        return dataset
    if expected:
        raise ValueError(
            f"{path} is missing, but the Bayesian checkpoint records "
            f"{expected} successful TraceWin evaluations."
        )
    return BeamDataset()


def _ensure_default_evaluation(
    report: dict,
    *,
    project_file: Path,
    calc_root: Path,
    timeout: float,
    retries: int,
    tracewin_particles: int,
    tracewin_threads: int | None,
) -> dict:
    """Evaluate the current adige.py default_params() with real TraceWin.

    Cached in the report so a resumed run does not re-evaluate the defaults.
    """
    if report.get("default_result") is not None:
        return report["default_result"]

    simulator = TraceWinSimulator(
        project_file=str(project_file),
        calc_dir=str(calc_root / "default_eval"),
        timeout=timeout,
        retries=retries,
        tracewin_params={"nbr_part1": tracewin_particles},
        num_threads=tracewin_threads,
        initial_npart=tracewin_particles,
    )
    print("\nEvaluating current adige.py defaults with TraceWin...", flush=True)
    params = default_params()
    result = simulator.simulate(params)
    score = float(result.score_val if result.success else ERROR_SCORE)
    status = "ok" if result.success else f"failed: {result.error}"
    print(f"  default score = {score:.6g} ({status})", flush=True)

    entry = {
        "params": params,
        "score": score,
        "success": bool(result.success),
        "error": result.error,
        "timestamp": result.timestamp.isoformat(),
    }
    report["default_result"] = entry
    return entry


def _build_optimizer(
    bounds: list[tuple[float, float]],
    *,
    optimizer_seed: int,
    warm_start: list[dict],
    evaluations: list[dict],
    initial_points: int = 0,
    initial_point_generator: str = "random",
):
    try:
        from skopt import Optimizer
        from skopt.space import Real
    except ImportError as exc:
        raise ImportError(
            "scikit-optimize is required: pip install scikit-optimize"
        ) from exc

    dimensions = [
        Real(lower, upper, name=key)
        for key, (lower, upper) in zip(PARAM_KEYS, bounds)
    ]
    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        n_initial_points=initial_points,
        initial_point_generator=initial_point_generator,
        acq_func="EI",
        random_state=optimizer_seed,
        avoid_duplicates=True,
    )
    penalty = _GPFailurePenalty(warm_start)
    if warm_start:
        x_values = [_vector_from_params(entry["params"]) for entry in warm_start]
        y_values = [-float(entry["score"]) for entry in warm_start]
        optimizer.tell(x_values, y_values)
    for evaluation in evaluations:
        # Replay ask() to restore Sobol/acquisition RNG state exactly on resume,
        # then tell() the persisted point and the same (penalized) objective the
        # live loop produced.
        optimizer.ask()
        optimizer.tell(
            _vector_from_params(evaluation["params"]),
            penalty.tell_value(
                float(evaluation["score"]),
                bool(evaluation.get("success", True)),
            ),
        )
    return optimizer


def _replayed_gp_penalty(warm_start: list[dict], evaluations: list[dict]) -> _GPFailurePenalty:
    """Rebuild the failure-penalty tracker in the same state _build_optimizer
    left it after replaying the persisted history."""
    penalty = _GPFailurePenalty(warm_start)
    for evaluation in evaluations:
        penalty.tell_value(
            float(evaluation["score"]),
            bool(evaluation.get("success", True)),
        )
    return penalty


def _best_entry(report: dict) -> dict | None:
    candidates = [
        {
            "origin": "warm_start",
            "run_index": None,
            "evaluation_index": None,
            "tracewin_seed": None,
            **entry,
        }
        for entry in report["warm_start"]
    ]
    candidates.extend(
        {
            "origin": "online_tracewin",
            "run_index": run["run_index"],
            **evaluation,
        }
        for run in report["runs"]
        for evaluation in run["evaluations"]
        if evaluation["success"]
    )
    if not candidates:
        return None
    return max(candidates, key=lambda entry: float(entry["score"]))


def _update_report_bests(report: dict) -> None:
    for run in report["runs"]:
        valid = [
            evaluation
            for evaluation in run["evaluations"]
            if evaluation["success"]
        ]
        if valid:
            best = max(valid, key=lambda entry: float(entry["score"]))
            run["best_score"] = float(best["score"])
            run["best_params"] = best["params"]
        else:
            run["best_score"] = None
            run["best_params"] = None
    report["best_result"] = _best_entry(report)
    report["updated_at"] = _utc_now()


def _save_datasets(
    source_dataset: BeamDataset | None,
    new_dataset: BeamDataset,
    *,
    new_samples_output: Path,
    merged_dataset_output: Path | None,
) -> None:
    new_dataset.save_flat(new_samples_output)
    if source_dataset is not None and merged_dataset_output is not None:
        source_dataset.merge(new_dataset).save_flat(merged_dataset_output)


def run_tracewin_bayesian(
    *,
    simulator_factory,
    source_dataset: BeamDataset | None,
    bounds: list[tuple[float, float]],
    report: dict,
    output: Path,
    new_samples_output: Path,
    merged_dataset_output: Path | None,
    n_calls: int,
    n_runs: int,
    seed: int,
    tracewin_seed_base: int | None,
    initial_points: int = 0,
    initial_point_generator: str = "random",
    random_tracewin_seeds: bool = False,
) -> dict:
    """Run or resume the ask/tell loop, persisting after every evaluation."""
    new_dataset = _load_new_samples(new_samples_output, report)
    _write_json_atomic(output, report)

    for run_index in range(n_runs):
        optimizer_seed = seed + run_index
        run = _ensure_run(report, run_index, optimizer_seed)
        optimizer = _build_optimizer(
            bounds,
            optimizer_seed=optimizer_seed,
            warm_start=report["warm_start"],
            evaluations=run["evaluations"],
            initial_points=initial_points,
            initial_point_generator=initial_point_generator,
        )
        gp_penalty = _replayed_gp_penalty(report["warm_start"], run["evaluations"])
        simulator = simulator_factory(run_index)

        while len(run["evaluations"]) < n_calls:
            call_index = len(run["evaluations"])
            evaluation_index = _completed_evaluation_count(report)
            if tracewin_seed_base is not None:
                tracewin_seed = tracewin_seed_base + evaluation_index
            elif random_tracewin_seeds:
                tracewin_seed = _random_unused_tracewin_seed(report)
            else:
                tracewin_seed = None

            if tracewin_seed is None:
                simulator.tracewin_params.pop("random_seed", None)
            else:
                simulator.tracewin_params["random_seed"] = int(tracewin_seed)

            vector = optimizer.ask()
            params = _params_from_vector(vector)
            print(
                f"\nRun {run_index + 1}/{n_runs} | "
                f"TraceWin call {call_index + 1}/{n_calls} | "
                f"evaluation={evaluation_index} "
                f"seed={tracewin_seed if tracewin_seed is not None else 'unset'}",
                flush=True,
            )
            result = simulator.simulate(params)
            score = float(result.score_val if result.success else ERROR_SCORE)
            optimizer.tell(vector, gp_penalty.tell_value(score, bool(result.success)))

            evaluation = {
                "evaluation_index": evaluation_index,
                "call_index": call_index,
                "phase": (
                    initial_point_generator
                    if call_index < initial_points
                    else "bayesian"
                ),
                "tracewin_seed": (
                    None if tracewin_seed is None else int(tracewin_seed)
                ),
                "params": params,
                "score": score,
                "success": bool(result.success),
                "error": result.error,
                "timestamp": result.timestamp.isoformat(),
            }
            run["evaluations"].append(evaluation)

            if result.success:
                x, y, sample_score = tracewin_result_to_flat_sample(result)
                new_dataset.append_flat_sample(x, y, sample_score)
                _save_datasets(
                    source_dataset,
                    new_dataset,
                    new_samples_output=new_samples_output,
                    merged_dataset_output=merged_dataset_output,
                )

            _update_report_bests(report)
            _write_json_atomic(output, report)
            status = "ok" if result.success else f"failed: {result.error}"
            print(f"  score={score:.6g} ({status})", flush=True)

        run["status"] = "complete"
        _update_report_bests(report)
        _write_json_atomic(output, report)

    report["status"] = "complete"
    _update_report_bests(report)
    _write_json_atomic(output, report)
    return report


def _print_best_params(best_result: dict, default_result: dict | None = None) -> None:
    if default_result is not None:
        status = "ok" if default_result["success"] else f"failed: {default_result['error']}"
        print(f"\nCurrent adige.py defaults: score={default_result['score']:.6g} ({status})")
        improvement = best_result["score"] - default_result["score"]
        print(f"Best vs. default         : {improvement:+.6g}")

    print("\nBest real TraceWin parameter set:")
    print(
        f"origin={best_result['origin']} score={best_result['score']:.6g} "
        f"seed={best_result.get('tracewin_seed')}"
    )
    header = (
        f"{'Parameter':<14} {'best':>14} {'default':>14} "
        f"{'delta/sensitivity':>18}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    sensitivities = sensitivity_vec()
    for index, parameter in enumerate(PARAMETERS):
        value = best_result["params"][parameter.key]
        delta_sensitivity = (
            (value - parameter.default) / sensitivities[index]
            if sensitivities[index] != 0
            else float("nan")
        )
        print(
            f"{parameter.name:<14} {value:>14.6g} "
            f"{parameter.default:>14.6g} {delta_sensitivity:>18.3g}"
        )
    print(separator)


def _format_delta_plot_label(normalized_delta: float, physical_delta: float) -> str:
    """Format the two delta units displayed on one parameter bar."""
    return f"{normalized_delta:+.3g} sens\nΔ={physical_delta:+.3g}"


def _delta_plot_score_summary(
    best_result: dict,
    default_result: dict | None = None,
) -> str:
    """Return the score subtitle shared by warm/cold delta plots."""
    best_score = float(best_result["score"])
    summary = f"Best score = {best_score:.6g}"
    if default_result is not None and bool(default_result.get("success", False)):
        default_score = float(default_result["score"])
        summary += (
            f" | Default score = {default_score:.6g}"
            f" | Δscore = {best_score - default_score:+.6g}"
        )
    return summary


def save_delta_plot(
    best_result: dict,
    output_json: str | Path,
    *,
    prefix: str = "bayesian_opt",
    default_result: dict | None = None,
) -> Path:
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sensitivities = sensitivity_vec()
    names = [parameter.name for parameter in PARAMETERS]
    physical_deltas = [
        float(best_result["params"][parameter.key]) - float(parameter.default)
        for parameter in PARAMETERS
    ]
    normalized_deltas = [
        physical_deltas[index] / sensitivities[index]
        for index, parameter in enumerate(PARAMETERS)
    ]
    figure, axis = plt.subplots(
        figsize=(max(12.0, 0.85 * len(names)), 6.8)
    )
    colors = [
        "#4c78a8" if delta >= 0 else "#e34948"
        for delta in normalized_deltas
    ]
    bars = axis.bar(names, normalized_deltas, color=colors, alpha=0.86)
    for bar, normalized_delta, physical_delta in zip(
        bars,
        normalized_deltas,
        physical_deltas,
    ):
        positive = normalized_delta >= 0.0
        axis.annotate(
            _format_delta_plot_label(normalized_delta, physical_delta),
            xy=(bar.get_x() + bar.get_width() / 2.0, normalized_delta),
            xytext=(0, 5 if positive else -5),
            textcoords="offset points",
            ha="center",
            va="bottom" if positive else "top",
            fontsize=7.2,
            rotation=90,
        )
    axis.axhline(0.0, color="#333333", linewidth=0.9)
    axis.set_title(
        "TraceWin Bayesian Optimization — shift from default\n"
        + _delta_plot_score_summary(best_result, default_result)
    )
    axis.set_ylabel("(best - default) / sensitivity")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=45)
    axis.margins(y=0.22)
    figure.tight_layout()
    path = Path(output_json).parent / f"{prefix}_delta.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_convergence_plot(
    report: dict,
    output_json: str | Path,
    *,
    prefix: str = "bayesian_opt",
) -> Path | None:
    histories = [
        [evaluation["score"] for evaluation in run["evaluations"]]
        for run in report["runs"]
        if run["evaluations"]
    ]
    if not histories:
        return None

    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    final_bests = []
    for run_index, history in enumerate(histories):
        best_so_far = np.maximum.accumulate(np.asarray(history, dtype=float))
        evaluations = np.arange(1, len(history) + 1)
        line, = axis.plot(
            evaluations,
            best_so_far,
            linewidth=1.8,
            label=f"run {run_index + 1}",
        )
        final_best = float(best_so_far[-1])
        final_bests.append(final_best)
        axis.scatter(
            evaluations[-1],
            final_best,
            s=46,
            color=line.get_color(),
            zorder=4,
        )
        axis.annotate(
            f"{final_best:.6g}",
            xy=(evaluations[-1], final_best),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )
    axis.set_xlabel("New TraceWin evaluations")
    axis.set_ylabel("Best observed score")
    axis.set_title(
        "TraceWin Bayesian Optimization convergence\n"
        f"Global best score = {max(final_bests):.6g}"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = Path(output_json).parent / f"{prefix}_convergence.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(default_dataset_path()))
    tracewin_source = parser.add_mutually_exclusive_group()
    tracewin_source.add_argument("--workspace", default=None, metavar="PATH")
    tracewin_source.add_argument(
        "--tracewin",
        default=None,
        metavar="INI",
        help=f"TraceWin project file (default: {DEFAULT_TRACEWIN_INI})",
    )
    parser.add_argument("--calc-dir", default=None, metavar="PATH")
    parser.add_argument("--n-calls", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--warm-best", type=int, default=10)
    parser.add_argument("--warm-diverse", type=int, default=30)
    parser.add_argument(
        "--bounds-scale",
        type=float,
        default=BAYESIAN_SCALE,
        help=(
            "Search half-width in sensitivity units around each default "
            "(default: BAYESIAN_SCALE from adige.py)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--new-samples-output", default=DEFAULT_NEW_SAMPLES_OUTPUT)
    parser.add_argument(
        "--merged-dataset-output",
        default=DEFAULT_MERGED_DATASET_OUTPUT,
    )
    args = parser.parse_args()

    if args.n_calls <= 0 or args.n_runs <= 0:
        parser.error("--n-calls and --n-runs must be positive")
    if args.warm_best < 0 or args.warm_diverse < 0:
        parser.error("--warm-best and --warm-diverse must be non-negative")

    try:
        workspace, project_file = resolve_tracewin_project(
            workspace=args.workspace,
            tracewin=args.tracewin,
        )
    except ValueError as exc:
        parser.error(str(exc))

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.is_file():
        parser.error(f"Dataset not found: {dataset_path}")
    source_dataset = BeamDataset.load(dataset_path)

    bounds = hardware_aware_bounds(PARAMETERS, args.bounds_scale)
    selection = select_warm_start(
        source_dataset.get_param_vecs().numpy(),
        source_dataset.scores.numpy(),
        parameters=PARAMETERS,
        bounds=bounds,
        n_best=args.warm_best,
        n_diverse=args.warm_diverse,
        seed=args.seed,
    )
    requested_warm = args.warm_best + args.warm_diverse
    if len(selection.indices) < requested_warm:
        print(
            f"WARNING: requested {requested_warm} warm-start rows, but only "
            f"{len(selection.indices)} valid unique rows are available.",
            flush=True,
        )

    output = Path(args.output).expanduser().resolve()
    calc_root = (
        Path(args.calc_dir).expanduser().resolve()
        if args.calc_dir
        else _default_calc_root(workspace, output)
    )
    new_samples_output = Path(args.new_samples_output).expanduser().resolve()
    merged_dataset_output = Path(args.merged_dataset_output).expanduser().resolve()
    config = _report_config(
        args,
        dataset_path=dataset_path,
        project_file=project_file,
        calc_root=calc_root,
        bounds=bounds,
    )
    warm_start = _warm_start_payload(selection)
    try:
        report = _load_or_create_report(
            output,
            config=config,
            warm_start=warm_start,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"TraceWin workspace : {workspace}")
    print(f"TraceWin project   : {project_file}")
    print(f"Dataset            : {dataset_path}")
    print(f"Warm start         : {len(warm_start)} known TraceWin points")
    print(f"New calls          : {args.n_calls} per run × {args.n_runs} run(s)")
    print(f"Bounds scale       : {args.bounds_scale} sensitivity units")
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
        source_dataset=source_dataset,
        bounds=bounds,
        report=report,
        output=output,
        new_samples_output=new_samples_output,
        merged_dataset_output=merged_dataset_output,
        n_calls=args.n_calls,
        n_runs=args.n_runs,
        seed=args.seed,
        tracewin_seed_base=args.tracewin_seed_base,
    )
    _print_best_params(report["best_result"], report.get("default_result"))
    convergence = save_convergence_plot(report, output)
    delta = save_delta_plot(
        report["best_result"],
        output,
        default_result=report.get("default_result"),
    )
    print(f"\nJSON checkpoint       -> {output}")
    print(f"New TraceWin samples  -> {new_samples_output}")
    print(f"Merged dataset        -> {merged_dataset_output}")
    if convergence is not None:
        print(f"Convergence plot      -> {convergence}")
    print(f"Delta plot            -> {delta}")


if __name__ == "__main__":
    main()
