"""Stress-test a trained RL policy with real TraceWin outside the dataset trust region."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from beam_optimization.algorithms import MODEL_FREE_ALGORITHMS, load_agent
from beam_optimization.config.adige import (
    ACTION_SCALE,
    ALL_PARTICLE_LOST_SCALE,
    MAX_STEPS,
    N_PARAMS,
    PARAM_KEYS,
    TEST_RESET_SCALE,
    action_bounds,
    clip_params_to_hw,
    default_params,
    params_to_vec,
    sensitivity_vec,
)
from beam_optimization.config.offline_utility.fail_scale_calculation import (
    physical_failure_message,
)
from beam_optimization.config.paths import (
    DEFAULT_FAIL_SCALE_BENCHMARK_OUTPUT,
    configure_matplotlib_cache,
    default_dataset_path,
    resolve_tracewin_project,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.scripts.common import select_eval_action


DEFAULT_EPISODES = 3
DEFAULT_MAX_RESET_ATTEMPTS = 96


def normalized_offsets(params: dict[str, float]) -> np.ndarray:
    defaults = default_params()
    return np.asarray(
        [(float(params[key]) - defaults[key]) for key in PARAM_KEYS],
        dtype=np.float64,
    ) / sensitivity_vec()


def shell_radius(params: dict[str, float]) -> float:
    """L-infinity parameter displacement in sensitivity units."""
    return float(np.max(np.abs(normalized_offsets(params))))


def optimum_distance(params: dict[str, float]) -> float:
    """RMS distance from configured optimum in sensitivity units."""
    offsets = normalized_offsets(params)
    return float(np.sqrt(np.mean(np.square(offsets))))


def sample_stress_params(
    rng: np.random.Generator,
    *,
    inner_scale: float,
    outer_scale: float,
    k_sigma: float = 3.0,
    max_draws: int = 100_000,
) -> dict[str, float]:
    """Draw from the outer Gaussian, accepting only the clipped 3-sigma shell."""
    if inner_scale <= 0.0 or outer_scale <= inner_scale:
        raise ValueError("outer_scale must be greater than positive inner_scale")
    defaults = default_params()
    default_vec = np.asarray([defaults[key] for key in PARAM_KEYS], dtype=np.float64)
    sensitivities = sensitivity_vec()
    inner_radius = k_sigma * float(inner_scale)
    outer_radius = k_sigma * float(outer_scale)
    for _ in range(max_draws):
        vector = default_vec + outer_scale * sensitivities * rng.standard_normal(N_PARAMS)
        params = clip_params_to_hw(
            {key: float(value) for key, value in zip(PARAM_KEYS, vector)}
        )
        radius = shell_radius(params)
        if inner_radius < radius <= outer_radius:
            return params
    raise RuntimeError(
        f"Could not sample a hardware-valid point in shell ({inner_radius}, {outer_radius}] "
        f"after {max_draws} draws"
    )


def trajectory_metrics(params: dict[str, float], dataset: BeamDataset) -> dict[str, float]:
    if len(dataset) < 5:
        raise ValueError("fail_scale_benchmark requires a dataset with at least 5 samples for KNN k=5")
    knn = float(dataset.param_knn_distance(params_to_vec(params), k=5)[0])
    return {
        "knn_distance": knn,
        "optimum_distance": optimum_distance(params),
        "shell_radius": shell_radius(params),
    }


def _result_fields(result) -> dict:
    return {
        "success": bool(result.success),
        "score": float(result.score_val),
        "error": result.error,
        "physics_failure": physical_failure_message(result),
    }


def _step_row(
    *,
    episode: int,
    reset_attempt: int,
    step: int,
    params: dict[str, float],
    result,
    dataset: BeamDataset,
    reward: float,
) -> dict:
    return {
        "episode": int(episode),
        "reset_attempt": int(reset_attempt),
        "step": int(step),
        "reward": float(reward),
        **_result_fields(result),
        **trajectory_metrics(params, dataset),
        "params": {key: float(value) for key, value in params.items()},
    }


def summarize_episode(rows: list[dict], termination: str) -> dict:
    initial = rows[0]
    final = rows[-1]
    successful_scores = [row["score"] for row in rows if row["success"]]
    return {
        "episode": initial["episode"],
        "reset_attempt": initial["reset_attempt"],
        "n_steps": len(rows) - 1,
        "termination": termination,
        "initial_score": initial["score"],
        "final_score": final["score"],
        "score_improvement": final["score"] - initial["score"],
        "best_successful_score": max(successful_scores) if successful_scores else None,
        "initial_knn_distance": initial["knn_distance"],
        "final_knn_distance": final["knn_distance"],
        "knn_reduction": initial["knn_distance"] - final["knn_distance"],
        "initial_optimum_distance": initial["optimum_distance"],
        "final_optimum_distance": final["optimum_distance"],
        "optimum_distance_reduction": (
            initial["optimum_distance"] - final["optimum_distance"]
        ),
        "minimum_optimum_distance": min(row["optimum_distance"] for row in rows),
    }


def run_fail_scale_benchmark(
    env,
    agent,
    dataset: BeamDataset,
    *,
    outer_scale: float,
    episodes: int = DEFAULT_EPISODES,
    max_reset_attempts: int = DEFAULT_MAX_RESET_ATTEMPTS,
    seed: int = 42,
    reference_result=None,
    verbose: bool = True,
) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_reset_attempts < episodes:
        raise ValueError("max_reset_attempts must be >= episodes")
    if outer_scale <= TEST_RESET_SCALE:
        raise ValueError("ALL_PARTICLE_LOST_SCALE must be greater than TEST_RESET_SCALE")

    if reference_result is None:
        reference_result = env.simulator.simulate(default_params())
    if not reference_result.success:
        raise RuntimeError(
            "TraceWin failed at default_params(); recovery reference is invalid: "
            f"{reference_result.error}"
        )

    rng = np.random.default_rng(seed)
    reset_attempt_rows: list[dict] = []
    episode_records: list[dict] = []
    valid_episodes = 0

    for attempt in range(1, max_reset_attempts + 1):
        if valid_episodes >= episodes:
            break
        params = sample_stress_params(
            rng,
            inner_scale=TEST_RESET_SCALE,
            outer_scale=outer_scale,
        )
        obs, info = env.reset(
            seed=seed + attempt - 1,
            options={"initial_params": params},
        )
        result = info["sim_result"]
        reset_record = {
            "reset_attempt": attempt,
            **_result_fields(result),
            **trajectory_metrics(env.current_params, dataset),
            "params": env.current_params,
        }
        reset_attempt_rows.append(reset_record)
        if verbose:
            print(
                f"reset attempt {attempt}/{max_reset_attempts} "
                f"success={result.success} score={result.score_val:.6g}",
                flush=True,
            )
        if not result.success:
            if physical_failure_message(result) is None:
                raise RuntimeError(
                    "Technical or unknown TraceWin reset failure; benchmark aborted: "
                    f"{result.error}"
                )
            continue

        valid_episodes += 1
        rows = [
            _step_row(
                episode=valid_episodes,
                reset_attempt=attempt,
                step=0,
                params=env.current_params,
                result=result,
                dataset=dataset,
                reward=0.0,
            )
        ]
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = select_eval_action(agent, obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            step_result = step_info["sim_result"]
            rows.append(
                _step_row(
                    episode=valid_episodes,
                    reset_attempt=attempt,
                    step=int(step_info["step"]),
                    params=env.current_params,
                    result=step_result,
                    dataset=dataset,
                    reward=reward,
                )
            )
            if verbose:
                print(
                    f"  episode={valid_episodes} step={step_info['step']} "
                    f"success={step_result.success} score={step_result.score_val:.6g}",
                    flush=True,
                )
        if terminated:
            termination = (
                "physics_failure"
                if rows[-1]["physics_failure"] is not None
                else "technical_or_unknown_failure"
            )
        else:
            termination = "max_steps"
        episode_records.append({
            "summary": summarize_episode(rows, termination),
            "trajectory": rows,
        })

    summaries = [record["summary"] for record in episode_records]
    completed = len(summaries)
    return {
        "reference": {
            **_result_fields(reference_result),
            "params": default_params(),
            **trajectory_metrics(default_params(), dataset),
        },
        "requested_episodes": int(episodes),
        "completed_valid_episodes": completed,
        "max_reset_attempts": int(max_reset_attempts),
        "reset_attempts_used": len(reset_attempt_rows),
        "reset_physics_failures": sum(not row["success"] for row in reset_attempt_rows),
        "reset_attempts": reset_attempt_rows,
        "episodes": episode_records,
        "summary": {
            "episodes_improving_score": sum(row["score_improvement"] > 0 for row in summaries),
            "episodes_reducing_knn": sum(row["knn_reduction"] > 0 for row in summaries),
            "episodes_approaching_optimum": sum(
                row["optimum_distance_reduction"] > 0 for row in summaries
            ),
            "target_completed": completed == episodes,
        },
    }


def save_episode_plot(record: dict, reference_score: float, output_dir: str | Path) -> Path:
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = record["trajectory"]
    steps = np.asarray([row["step"] for row in rows])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))

    axes[0].plot(steps, [row["knn_distance"] for row in rows], marker="o")
    axes[0].set_title("KNN distance from dataset")
    axes[0].set_ylabel("mean standardized 5-NN distance")

    successful = [row for row in rows if row["success"]]
    axes[1].plot(
        [row["step"] for row in successful],
        [row["score"] for row in successful],
        marker="o",
        label="policy trajectory",
    )
    axes[1].axhline(reference_score, color="tab:green", linestyle="--", label="default optimum")
    failed_steps = [row["step"] for row in rows if not row["success"]]
    if failed_steps:
        y_min = min([row["score"] for row in successful] + [reference_score])
        axes[1].scatter(failed_steps, [y_min] * len(failed_steps), marker="x", color="tab:red", label="physics failure")
    axes[1].set_title("TraceWin score")
    axes[1].set_ylabel("score (higher is better)")
    axes[1].legend(fontsize=8)

    axes[2].plot(steps, [row["optimum_distance"] for row in rows], marker="o")
    axes[2].axhline(0.0, color="tab:green", linestyle="--", label="default optimum")
    axes[2].set_title("Distance from optimized defaults")
    axes[2].set_ylabel("RMS displacement / sensitivity")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("environment step")
        ax.grid(alpha=0.25)
    summary = record["summary"]
    fig.suptitle(
        f"Fail-scale recovery | episode {summary['episode']} | {summary['termination']}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output_dir / f"fail_scale_benchmark_episode_{summary['episode']:03d}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_csvs(report: dict, output_json: str | Path) -> dict[str, str]:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    reset_path = output_json.with_name(f"{output_json.stem}_reset_attempts.csv")
    steps_path = output_json.with_name(f"{output_json.stem}_steps.csv")

    reset_fields = [
        "reset_attempt", "success", "score", "error", "physics_failure",
        "knn_distance", "optimum_distance", "shell_radius", "params",
    ]
    with reset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reset_fields)
        writer.writeheader()
        for row in report["reset_attempts"]:
            writer.writerow({**row, "params": json.dumps(row["params"], sort_keys=True)})

    step_fields = [
        "episode", "reset_attempt", "step", "reward", "success", "score",
        "error", "physics_failure", "knn_distance", "optimum_distance",
        "shell_radius", "params",
    ]
    with steps_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=step_fields)
        writer.writeheader()
        for episode in report["episodes"]:
            for row in episode["trajectory"]:
                writer.writerow({**row, "params": json.dumps(row["params"], sort_keys=True)})
    return {"reset_attempts": str(reset_path), "steps": str(steps_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workspace", default=None, metavar="PATH")
    source.add_argument("--tracewin", default=None, metavar="INI")
    parser.add_argument("--calc-dir", default=None, metavar="PATH")
    parser.add_argument("--algo", choices=MODEL_FREE_ALGORITHMS, default="sac")
    parser.add_argument("--policy", required=True, metavar="CKPT")
    parser.add_argument("--dataset", default=str(default_dataset_path()), metavar="PT")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-reset-attempts", type=int, default=DEFAULT_MAX_RESET_ATTEMPTS)
    parser.add_argument("--max-ep-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_FAIL_SCALE_BENCHMARK_OUTPUT),
        metavar="JSON",
    )
    parser.add_argument("--plots-dir", default=None, metavar="PATH")
    args = parser.parse_args()

    if ALL_PARTICLE_LOST_SCALE is None:
        parser.error(
            "ALL_PARTICLE_LOST_SCALE is unset; run fail_scale_calculation --update-config first"
        )
    policy_path = Path(args.policy).expanduser().resolve()
    if not policy_path.is_file():
        parser.error(f"Policy checkpoint not found: {policy_path}")
    try:
        workspace, project_file = resolve_tracewin_project(
            workspace=args.workspace, tracewin=args.tracewin
        )
    except ValueError as exc:
        parser.error(str(exc))

    from beam_optimization.env.tracewin_env import TraceWinEnv

    calc_dir = (
        Path(args.calc_dir).expanduser().resolve()
        if args.calc_dir
        else workspace / "fail_scale_benchmark_calc"
    )
    dataset = BeamDataset.load(args.dataset)
    env = TraceWinEnv(
        project_file=str(project_file),
        calc_dir=str(calc_dir),
        max_steps=args.max_ep_steps,
        timeout=args.timeout,
        retries=args.retries,
        reset_scale=float(ALL_PARTICLE_LOST_SCALE),
    )
    low, high = action_bounds()
    agent = load_agent(
        args.algo,
        str(policy_path),
        env.observation_space.shape[0],
        N_PARAMS,
        (low.tolist(), high.tolist()),
        hidden_dims=args.hidden,
    )
    try:
        report = run_fail_scale_benchmark(
            env,
            agent,
            dataset,
            outer_scale=float(ALL_PARTICLE_LOST_SCALE),
            episodes=args.episodes,
            max_reset_attempts=args.max_reset_attempts,
            seed=args.seed,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    output = Path(args.output).expanduser().resolve()
    plots_dir = (
        Path(args.plots_dir).expanduser().resolve()
        if args.plots_dir
        else output.with_name(f"{output.stem}_plots")
    )
    plot_paths = [
        str(save_episode_plot(record, report["reference"]["score"], plots_dir))
        for record in report["episodes"]
    ]
    csv_paths = save_csvs(report, output)
    payload = {
        "config": {
            "algorithm": args.algo,
            "policy": str(policy_path),
            "dataset": str(Path(args.dataset).expanduser().resolve()),
            "workspace": str(workspace),
            "project": str(project_file),
            "calc_dir": str(calc_dir),
            "test_reset_scale": TEST_RESET_SCALE,
            "all_particle_lost_scale": ALL_PARTICLE_LOST_SCALE,
            "action_scale": ACTION_SCALE,
            "shell_k_sigma": 3.0,
            "episodes": args.episodes,
            "max_reset_attempts": args.max_reset_attempts,
            "max_ep_steps": args.max_ep_steps,
            "seed": args.seed,
        },
        **report,
        "csv": csv_paths,
        "plots": plot_paths,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nFAIL-SCALE BENCHMARK SUMMARY")
    print(f"  valid episodes:       {report['completed_valid_episodes']}/{args.episodes}")
    print(f"  reset attempts:       {report['reset_attempts_used']}")
    print(f"  reset physics losses: {report['reset_physics_failures']}")
    print(f"  approach optimum:     {report['summary']['episodes_approaching_optimum']}")
    print(f"  improve score:        {report['summary']['episodes_improving_score']}")
    print(f"  JSON:                 {output}")


if __name__ == "__main__":
    main()
