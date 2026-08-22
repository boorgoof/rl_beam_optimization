"""
Benchmark — compares optimization methods on the surrogate.

All methods receive the same surrogate, the same initial beam0 and the same
evaluation budget.

Methods:
  bayesian_opt      Bayesian Optimization (GP), a classical black-box
                    baseline, not a trained checkpoint. Skipped with
                    --policy-only.
  svg_final         SVGAgent -- final-stage reward only. Only trained from
  svg_uniform       scratch as a comparison method if --svg-baseline is given
                    (off by default: unlike BO, SVG is a full RL method, not
                    a classical baseline). To evaluate an SVG agent you
                    already trained (e.g. via train_policies), pass it via
                    --svg-final/--svg-uniform instead, like any other
                    trained policy.
  trained policies  Any checkpoint passed via --sac/--sac-custom/.../--svg-uniform
                    is evaluated over --policy-episodes independent episodes.

Usage:
    python -m beam_optimization benchmark \\
        --surrogate env/surrogate_env/surrogate/trained_models/base/surrogate_001_0.pt \\
        --dataset   env/dataset/001/dataset_all.pt \\
        --output    results/benchmark.json \\
        --n-runs    3 \\
        --eval-budget 3000

With trained agents:
    python -m beam_optimization benchmark \\
        --sac results/train/rl/all/sac/sac_agent.zip \\
        --td3-custom results/train/rl/all/td3_custom/td3_custom_agent.pt \\
        --ppo results/train/rl/all/ppo/ppo_agent.zip

Also train the from-scratch SVG comparison baseline:
    python -m beam_optimization benchmark --svg-baseline --svg-episodes 500

Quick smoke test:
    python -m beam_optimization benchmark --quick
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from beam_optimization.algorithms import (
    CUSTOM_MODEL_FREE_ALGORITHMS,
    STABLE_BASELINES_ALGORITHMS,
    canonical_algorithm_name,
    make_custom_agent,
)
from beam_optimization.config.paths import (
    DEFAULT_BENCHMARK_OUTPUT,
    DEFAULT_TRACEWIN_INI,
    configure_matplotlib_cache,
    default_dataset_path,
    default_single_surrogate_model,
)
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.scripts.common import algo_style, run_episode
from beam_optimization.config.adige import (
    BAYESIAN_SCALE,
    MAX_STEPS,
    N_OUTPUT_STAGES,
    N_PARAMS,
    PARAM_KEYS,
    PARAMETERS,
    TEST_RESET_SCALE,
    action_bounds,
    default_params,
    observation_dim,
    params_to_stage_tensors,
    BEAM_STATE_FEATURES,
    score,
    score_function_metadata,
)

OBS_DIM = observation_dim()
ACT_DIM = N_PARAMS
ACTION_BOUNDS = tuple(v.tolist() for v in action_bounds())
POLICY_ALGORITHMS = (
    *STABLE_BASELINES_ALGORITHMS,
    *CUSTOM_MODEL_FREE_ALGORITHMS,
    "mbpo",
    "iterative_sim2real_sac",
    "svg_final",
    "svg_uniform",
)

# skopt refits the GP after every evaluation, so its cost grows quickly with
# the number of calls; cap the BO baseline here even when --eval-budget is larger.
BO_MAX_CALLS = 200

STAGE_WEIGHT_CONFIGS = {
    "final":   None,
    "uniform": [1.0] * N_OUTPUT_STAGES,
}


# ── Benchmark functions ───────────────────────────────────────────────────────


def run_bo(surrogate, dataset, budget, seed) -> Dict:
    from beam_optimization.algorithms.baselines.bayesian_opt import (
        BayesianOptimizer,
        hardware_aware_bounds,
    )
    beam0 = _pick_beam(dataset, seed)
    surrogate.eval()
    def objective(params):
        with torch.no_grad():
            stage_tensors = params_to_stage_tensors(params)
            outs = surrogate(stage_tensors, beam0)
            final_beam = {v: float(outs[-1][0, i]) for i, v in enumerate(BEAM_STATE_FEATURES)}
            return score(final_beam)

    # Same search box as everywhere else in the project (default ±
    # BAYESIAN_SCALE·sensitivity, clipped to hardware): the surrogate is only
    # trained inside this trust region, so searching wider would let BO score
    # extrapolation artifacts instead of physics.
    result = BayesianOptimizer(
        n_calls=min(budget, BO_MAX_CALLS),
        seed=seed,
        param_keys=PARAM_KEYS,
        bounds=hardware_aware_bounds(PARAMETERS, BAYESIAN_SCALE),
    ).optimize(objective)
    return {
        "best_score": result.best_score,
        "history": result.score_history,
        "best_params": result.best_params,
    }


def run_svg(surrogate, dataset, n_episodes, horizon, seed, stage_weights) -> Dict:
    from beam_optimization.algorithms.model_based.svg import SVGAgent
    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    agent = SVGAgent(
        surrogate=surrogate,
        dataset=dataset,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        action_bounds=ACTION_BOUNDS,
        param_keys=PARAM_KEYS,
        default_params=default_params(),
        n_step=horizon,
        stage_weights=stage_weights,
    )
    history = []
    for ep in range(n_episodes):
        result = agent.optimize_episode()
        history.append(result.final_score)
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"    ep {ep+1}/{n_episodes}  score={result.final_score:.3f}")

    return {"best_score": float(max(history)), "history": history}


# ── Final policy benchmark ────────────────────────────────────────────────────

def make_policy_agent(algo: str, ckpt_path: str, env, hidden: list[int],
                      surrogate=None, dataset=None):
    """Instantiate and load a trained policy for deterministic evaluation.

    surrogate/dataset are needed only to rebuild SVG agents (their policy is
    env-independent at evaluation time, so they can also be scored on TraceWin).
    """
    bounds = ACTION_BOUNDS
    obs_dim = env.observation_space.shape[0]

    algo = canonical_algorithm_name(algo)
    if algo == "iterative_sim2real_sac":
        from beam_optimization.algorithms.model_free.stable_baselines import (
            StableBaselinesAgent,
        )
        return StableBaselinesAgent.load("sac", ckpt_path, env=env)
    if algo in STABLE_BASELINES_ALGORITHMS:
        from beam_optimization.algorithms.model_free.stable_baselines import (
            StableBaselinesAgent,
        )
        return StableBaselinesAgent.load(algo, ckpt_path, env=env)
    if algo in {"svg_final", "svg_finale", "svg_uniform"}:
        from beam_optimization.algorithms.model_based.svg import SVGAgent
        stage_weights = STAGE_WEIGHT_CONFIGS["uniform" if algo == "svg_uniform" else "final"]
        agent = SVGAgent(
            surrogate=surrogate,
            dataset=dataset,
            obs_dim=obs_dim,
            act_dim=ACT_DIM,
            action_bounds=bounds,
            param_keys=PARAM_KEYS,
            default_params=default_params(),
            hidden_dims=tuple(hidden),
            n_step=env.max_steps,
            stage_weights=stage_weights,
        )
    else:
        # "mbpo" checkpoints are the inner SAC saved by train_dyna.
        name = "sac_custom" if algo == "mbpo" else algo
        agent = make_custom_agent(
            name, obs_dim, ACT_DIM, bounds, hidden_dims=hidden
        )

    agent.load(ckpt_path)
    return agent


def run_policy_episode(env, agent, algo: str, seed: int, episode_idx: int) -> dict:
    result = run_episode(env, agent, seed=seed)
    features = result["final_features"]
    final_ex = float(features.get("ex", np.nan))
    final_ey = float(features.get("ey", np.nan))
    best_params = {
        key: float(env.state.best_params[key])
        for key in PARAM_KEYS
    }
    return {
        "algorithm": algo,
        "episode": int(episode_idx),
        "total_reward": result["total_reward"],
        "final_score": result["final_score"],
        "final_ex": final_ex,
        "final_ey": final_ey,
        "final_emittance": float((final_ex + final_ey) / 2.0),
        "final_npart_ratio": float(features.get("npart_ratio", np.nan)),
        "n_steps": result["n_steps"],
        # Private aggregation fields: summarize_policy_episodes() promotes only
        # the global winner to the public summary. They are removed before the
        # per-episode JSON/CSV is persisted.
        "_best_observed_score": float(env.state.best_score),
        "_best_observed_step": int(env.state.best_step),
        "_best_observed_params": best_params,
    }


def summarize_policy_episodes(episodes: list[dict]) -> dict[str, dict]:
    metrics = ["total_reward", "final_score", "final_emittance", "final_npart_ratio"]
    summary: dict[str, dict] = {}
    algorithms = sorted({row["algorithm"] for row in episodes})
    for algo in algorithms:
        rows = [row for row in episodes if row["algorithm"] == algo]
        algo_summary = {"episodes": len(rows)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            algo_summary[f"{metric}_mean"] = float(np.nanmean(values))
            algo_summary[f"{metric}_std"] = float(np.nanstd(values))
        # Strict comparison preserves the first occurrence when scores tie.
        winner = rows[0]
        for candidate in rows[1:]:
            if candidate["_best_observed_score"] > winner["_best_observed_score"]:
                winner = candidate
        algo_summary.update({
            "best_observed_score": float(winner["_best_observed_score"]),
            "best_observed_episode": int(winner["episode"]),
            "best_observed_step": int(winner["_best_observed_step"]),
            "best_observed_params": {
                key: float(winner["_best_observed_params"][key])
                for key in PARAM_KEYS
            },
        })
        summary[algo] = algo_summary
    return summary


def print_policy_table(summary: dict[str, dict]) -> None:
    print(f"\n{'='*112}")
    print("FINAL POLICY BENCHMARK")
    print(
        f"{'Algorithm':<18} {'Reward mean±std':>22} {'Score mean±std':>22} "
        f"{'Emit mean±std':>22} {'Npart mean±std':>22}"
    )
    print("-" * 112)
    for algo, row in sorted(summary.items()):
        print(
            f"{algo:<18} "
            f"{row['total_reward_mean']:>9.3f}±{row['total_reward_std']:<9.3f} "
            f"{row['final_score_mean']:>9.3f}±{row['final_score_std']:<9.3f} "
            f"{row['final_emittance_mean']:>9.5f}±{row['final_emittance_std']:<9.5f} "
            f"{row['final_npart_ratio_mean']:>9.5f}±{row['final_npart_ratio_std']:<9.5f}"
        )
    print(f"{'='*112}")


def write_policy_csvs(episodes: list[dict], summary: dict[str, dict], output_json: str | Path,
                      tag: str = "") -> tuple[Path, Path]:
    out_dir = Path(output_json).parent
    episodes_path = out_dir / f"benchmark_policy_episodes{tag}.csv"
    summary_path = out_dir / f"benchmark_policy_summary{tag}.csv"

    episode_fields = [
        "algorithm",
        "episode",
        "total_reward",
        "final_score",
        "final_ex",
        "final_ey",
        "final_emittance",
        "final_npart_ratio",
        "n_steps",
    ]
    with open(episodes_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_fields)
        writer.writeheader()
        writer.writerows(episodes)

    summary_fields = [
        "algorithm",
        "episodes",
        "total_reward_mean",
        "total_reward_std",
        "final_score_mean",
        "final_score_std",
        "final_emittance_mean",
        "final_emittance_std",
        "final_npart_ratio_mean",
        "final_npart_ratio_std",
        "best_observed_score",
        "best_observed_episode",
        "best_observed_step",
        *[f"best_param_{key}" for key in PARAM_KEYS],
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for algo, row in sorted(summary.items()):
            csv_row = {
                key: value
                for key, value in row.items()
                if key != "best_observed_params"
            }
            csv_row.update({
                f"best_param_{key}": row["best_observed_params"][key]
                for key in PARAM_KEYS
            })
            writer.writerow({"algorithm": algo, **csv_row})

    return episodes_path, summary_path


# Panels of the policy benchmark figures: (metric key, title, y-label).
POLICY_PANELS = [
    ("final_score", "Final score", "score (higher is better)"),
    ("final_emittance", "Final emittance", "(ex + ey) / 2 (lower is better)"),
    ("final_npart_ratio", "Final particle ratio", "npart ratio (higher is better)"),
    ("total_reward", "Cumulative RL reward", "Σ bounded absolute reward"),
]


def _optimization_best(results: Optional[dict], method: str) -> Optional[float]:
    """Mean best score across runs for one optimization method, if present."""
    runs = (results or {}).get(method)
    if not runs:
        return None
    return float(np.mean([r["best_score"] for r in runs]))


def save_policy_plots(episodes: list[dict], summary: dict[str, dict], output_json: str | Path,
                      tag: str = "",
                      optimization_results: Optional[dict] = None) -> tuple[Path, Path]:
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = sorted(summary)
    colors = [algo_style(algo)[0] for algo in algorithms]
    out_dir = Path(output_json).parent

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), POLICY_PANELS):
        means = [summary[algo][f"{metric}_mean"] for algo in algorithms]
        stds = [summary[algo][f"{metric}_std"] for algo in algorithms]
        ax.bar(algorithms, means, yerr=stds, capsize=4, alpha=0.86, color=colors)
        if metric == "final_score":
            for ref in ("bayesian_opt",):
                best = _optimization_best(optimization_results, ref)
                if best is not None:
                    ax.axhline(best, color=algo_style(ref)[0], linewidth=1.2,
                               linestyle=":", label=f"{ref} best")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    bar_path = out_dir / f"benchmark_policy_bars{tag}.png"
    fig.savefig(bar_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), POLICY_PANELS):
        values = [
            [row[metric] for row in episodes if row["algorithm"] == algo]
            for algo in algorithms
        ]
        try:
            boxes = ax.boxplot(values, tick_labels=algorithms, showmeans=True, patch_artist=True)
        except TypeError:
            boxes = ax.boxplot(values, labels=algorithms, showmeans=True, patch_artist=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    box_path = out_dir / f"benchmark_policy_boxplots{tag}.png"
    fig.savefig(box_path, dpi=160)
    plt.close(fig)

    return bar_path, box_path


def run_policy_benchmark(args, surrogate, dataset,
                         env_factory: Optional[Callable] = None,
                         episodes: Optional[int] = None,
                         tag: str = "",
                         optimization_results: Optional[dict] = None) -> dict:
    """Evaluate trained policy checkpoints over independent episodes.

    By default policies run on SurrogateEnv; pass env_factory (e.g. TraceWinEnv)
    and a smaller `episodes` for real-physics validation. `tag` suffixes the
    CSV/plot filenames (e.g. "_tracewin").
    """
    if env_factory is None:
        env_factory = lambda: SurrogateEnv(
            model=surrogate, dataset=dataset, max_steps=args.max_ep_steps,
            reset_scale=TEST_RESET_SCALE,
        )
    if episodes is None:
        episodes = args.policy_episodes

    checkpoint_args = {
        algo: getattr(args, algo, None)
        for algo in POLICY_ALGORITHMS
    }
    legacy_sb3_sac = getattr(args, "sb3_sac", None)
    if legacy_sb3_sac is not None:
        if checkpoint_args["sac"] is not None and checkpoint_args["sac"] != legacy_sb3_sac:
            raise ValueError("Pass only one of 'sac' and legacy 'sb3_sac'.")
        warnings.warn(
            "'sb3_sac' is deprecated; use 'sac'.",
            DeprecationWarning,
            stacklevel=2,
        )
        checkpoint_args["sac"] = legacy_sb3_sac
    checkpoints = {
        algo: Path(path)
        for algo, path in checkpoint_args.items()
        if path is not None
    }
    if not checkpoints:
        return {}

    missing = {algo: path for algo, path in checkpoints.items() if not path.exists()}
    if missing:
        details = ", ".join(f"{algo}: {path}" for algo, path in missing.items())
        raise FileNotFoundError(f"Policy checkpoint not found: {details}")
    wrong_suffix = {}
    for algo, path in checkpoints.items():
        expected = (
            ".zip"
            if algo in STABLE_BASELINES_ALGORITHMS
            or algo == "iterative_sim2real_sac"
            else ".pt"
        )
        if path.suffix.lower() != expected:
            wrong_suffix[algo] = (path, expected)
    if wrong_suffix:
        details = ", ".join(
            f"{algo}: expected {expected}, got {path.name}"
            for algo, (path, expected) in wrong_suffix.items()
        )
        raise ValueError(f"Invalid policy checkpoint extension: {details}")

    all_episodes: list[dict] = []
    label = tag.lstrip("_") or "surrogate"
    print(f"\n{'='*65}\nFinal policy benchmark [{label}] ({episodes} episodes)\n{'='*65}")
    for algo, ckpt_path in sorted(checkpoints.items()):
        print(f"{algo}: {ckpt_path}")
        env = env_factory()
        agent = make_policy_agent(algo, str(ckpt_path), env, args.hidden,
                                  surrogate=surrogate, dataset=dataset)
        try:
            for episode_idx in range(episodes):
                seed = args.policy_seed + episode_idx
                row = run_policy_episode(env, agent, algo, seed, episode_idx)
                all_episodes.append(row)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    summary = summarize_policy_episodes(all_episodes)
    public_episodes = [
        {
            key: value
            for key, value in row.items()
            if not key.startswith("_best_observed_")
        }
        for row in all_episodes
    ]
    print_policy_table(summary)
    episodes_csv, summary_csv = write_policy_csvs(
        public_episodes, summary, args.output, tag=tag
    )
    print(f"Policy episode CSV saved → {episodes_csv}")
    print(f"Policy summary CSV saved → {summary_csv}")

    plot_paths = {}
    if not args.no_policy_plots:
        bar_path, box_path = save_policy_plots(
            public_episodes, summary, args.output,
            tag=tag, optimization_results=optimization_results,
        )
        plot_paths = {"bar_plot": str(bar_path), "box_plot": str(box_path)}
        print(f"Policy bar plot saved → {bar_path}")
        print(f"Policy boxplot saved  → {box_path}")

    return {
        "episodes": public_episodes,
        "summary": summary,
        "csv": {
            "episodes": str(episodes_csv),
            "summary": str(summary_csv),
        },
        "plots": plot_paths,
    }


# ── Utility ───────────────────────────────────────────────────────────────────

def _pick_beam(dataset, seed) -> torch.Tensor:
    idx = int(np.random.default_rng(seed).integers(0, len(dataset.get_initial_beam_states())))
    return dataset.get_initial_beam_states()[idx:idx+1].float()


def print_table(results: Dict):
    print(f"\n{'='*65}")
    print("BENCHMARK SUMMARY")
    print(f"{'Method':<35}  {'Mean':>8}  {'Std':>7}  {'Best':>8}")
    print("-" * 65)
    for method, runs in sorted(results.items()):
        bests = [r["best_score"] for r in runs]
        print(f"{method:<35}  {np.mean(bests):>8.3f}  {np.std(bests):>7.3f}  {np.max(bests):>8.3f}")
    print(f"{'='*65}")


def _best_so_far(history) -> np.ndarray:
    """Cumulative maximum of a score history."""
    return np.maximum.accumulate(np.asarray(history, dtype=float))


def save_convergence_plot(results: Dict, output_json: str | Path,
                          svg_horizon: int) -> Optional[Path]:
    """Sample-efficiency plot: best score so far vs surrogate evaluations.

    BO histories contain one entry per objective evaluation; SVG histories
    one entry per episode (= svg_horizon surrogate calls). Mean across the
    n_runs with a ±std band, truncated to the shortest run.
    """
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    plotted = False
    for method, runs in sorted(results.items()):
        histories = [r["history"] for r in runs if r.get("history")]
        if not histories:
            continue
        n = min(len(h) for h in histories)
        curves = np.stack([_best_so_far(h[:n]) for h in histories])
        evals_per_point = svg_horizon if method.startswith("svg") else 1
        x = np.arange(1, n + 1) * evals_per_point
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        color, linestyle = algo_style(method)
        # Short histories (few points) would be invisible as a bare line.
        marker = "o" if n < 25 else None
        ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.8,
                marker=marker, markersize=4, label=method)
        if curves.shape[0] > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xscale("log")
    ax.set_xlabel("Surrogate evaluations (log scale)")
    ax.set_ylabel("Best score so far (higher is better)")
    ax.set_title("Sample efficiency")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()

    path = Path(output_json).parent / "benchmark_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_summary_plot(results: Dict, output_json: str | Path) -> Path:
    """Save a bar chart comparing benchmark best scores."""
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for method, runs in sorted(results.items()):
        bests = np.asarray([r["best_score"] for r in runs], dtype=float)
        rows.append(
            {
                "method": method,
                "mean": float(np.mean(bests)),
                "std": float(np.std(bests)),
                "best": float(np.max(bests)),
            }
        )
    rows.sort(key=lambda row: row["mean"], reverse=True)

    methods = [row["method"] for row in rows]
    means = [row["mean"] for row in rows]
    stds = [row["std"] for row in rows]

    fig_width = max(8.0, 0.9 * len(methods))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    colors = [algo_style(method)[0] for method in methods]
    bars = ax.bar(methods, means, yerr=stds, color=colors, alpha=0.86, capsize=4)

    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_title("Benchmark comparison")
    ax.set_ylabel("Mean best score across runs")
    ax.set_xlabel("Method")
    ax.text(
        0.99,
        0.98,
        "Higher is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)

    y_min, y_max = ax.get_ylim()
    offset = 0.025 * (y_max - y_min)
    for bar, row in zip(bars, rows):
        value = row["mean"]
        label_y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.1f}",
            ha="center",
            va=va,
            fontsize=8,
        )

    fig.tight_layout()
    plot_path = Path(output_json).with_suffix(".png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate",    default=str(default_single_surrogate_model()))
    parser.add_argument("--dataset",      default=str(default_dataset_path()))
    parser.add_argument("--output",       default=str(DEFAULT_BENCHMARK_OUTPUT))
    parser.add_argument("--n-runs",       type=int, default=3)
    parser.add_argument("--eval-budget",  type=int, default=3000)
    parser.add_argument("--svg-episodes", type=int, default=500)
    parser.add_argument("--svg-horizon",  type=int, default=MAX_STEPS)
    parser.add_argument("--svg-baseline", action="store_true",
                        help="Also train a fresh SVGAgent from scratch (both svg_final/"
                             "svg_uniform stage-weight variants, --svg-episodes each) as a "
                             "comparison method, in addition to Bayesian Optimization. Off "
                             "by default: unlike BO, SVG is a full RL method, not a classical "
                             "baseline -- to evaluate an SVG agent you already trained "
                             "(e.g. via train_policies), use --svg-final/--svg-uniform "
                             "instead, exactly like --sac/--td3/etc.")
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help=(
            "Benchmark only supplied policy checkpoints: skip Bayesian "
            "Optimization and the optional from-scratch SVG baselines."
        ),
    )
    parser.add_argument(
        "--tracewin-only",
        action="store_true",
        help=(
            "Skip the surrogate policy benchmark and evaluate supplied "
            "checkpoints only on TraceWin. Requires --tracewin."
        ),
    )
    parser.add_argument("--policy-episodes", type=int, default=50,
                        help="Independent episodes per trained policy in the final policy benchmark.")
    parser.add_argument("--max-ep-steps", type=int, default=MAX_STEPS,
                        help="Max environment steps per policy-evaluation episode.")
    parser.add_argument("--policy-seed", type=int, default=42,
                        help="Base seed for final policy benchmark episodes.")
    parser.add_argument("--tracewin", default=None, metavar="INI",
                        nargs="?", const=str(DEFAULT_TRACEWIN_INI),
                        help="Also validate the passed policy checkpoints on the real "
                             "TraceWin environment (~30 s per step). Without a value, "
                             "uses the project default .ini.")
    parser.add_argument("--tracewin-episodes", type=int, default=5,
                        help="Episodes per policy in the TraceWin validation "
                             "(default: 5; keep small — real physics is slow).")
    parser.add_argument(
        "--no-kill-stale",
        action="store_true",
        help=(
            "Do not perform global TraceWin/Xvfb cleanup during TraceWin "
            "validation. Use this when other independent TraceWin workspaces "
            "are running concurrently."
        ),
    )
    parser.add_argument(
        "--tracewin-threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Threads used by each TraceWin validation simulation. Omitted: "
            "all CPUs. Set an explicit share for concurrent workspaces."
        ),
    )
    parser.add_argument(
        "--tracewin-timeout",
        type=float,
        default=45.0,
        metavar="SECONDS",
        help="Timeout for each TraceWin validation simulation (default: 45).",
    )
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer sizes used to recreate checkpointed custom agents.")
    parser.add_argument("--no-policy-plots", action="store_true",
                        help="Disable policy bar plot and boxplot generation.")
    parser.add_argument("--quick",        action="store_true")
    parser.add_argument("--sac",          default=None, metavar="CKPT", help="SB3 SAC .zip")
    parser.add_argument("--ppo",          default=None, metavar="CKPT", help="SB3 PPO .zip")
    parser.add_argument("--td3",          default=None, metavar="CKPT", help="SB3 TD3 .zip")
    parser.add_argument("--ddpg",         default=None, metavar="CKPT", help="SB3 DDPG .zip")
    parser.add_argument("--a2c",          default=None, metavar="CKPT", help="SB3 A2C .zip")
    parser.add_argument("--sac-custom",       dest="sac_custom", default=None, metavar="CKPT")
    parser.add_argument("--ppo-custom",       dest="ppo_custom", default=None, metavar="CKPT")
    parser.add_argument("--td3-custom",       dest="td3_custom", default=None, metavar="CKPT")
    parser.add_argument("--ddpg-custom",      dest="ddpg_custom", default=None, metavar="CKPT")
    parser.add_argument("--a2c-custom",       dest="a2c_custom", default=None, metavar="CKPT")
    parser.add_argument("--reinforce-custom", dest="reinforce_custom", default=None, metavar="CKPT")
    parser.add_argument("--trpo-custom",      dest="trpo_custom", default=None, metavar="CKPT")
    parser.add_argument(
        "--sb3-sac", dest="sb3_sac", default=None, metavar="CKPT",
        help="Deprecated alias for --sac.",
    )
    parser.add_argument("--mbpo",         default=None, metavar="CKPT")
    parser.add_argument(
        "--iterative-sim2real-sac",
        dest="iterative_sim2real_sac",
        default=None,
        metavar="CKPT",
        help="Iterative Sim-to-Real SAC checkpoint (.zip).",
    )
    parser.add_argument(
        "--svg-final", "--svg-finale", dest="svg_final", default=None, metavar="CKPT",
        help="SVG final-stage checkpoint (--svg-finale is a deprecated alias).",
    )
    parser.add_argument("--svg-uniform",  dest="svg_uniform", default=None, metavar="CKPT")
    args = parser.parse_args()
    if args.tracewin_threads is not None and args.tracewin_threads <= 0:
        parser.error("--tracewin-threads must be positive")
    if args.tracewin_timeout <= 0.0:
        parser.error("--tracewin-timeout must be positive")

    if args.tracewin_only and not args.tracewin:
        parser.error("--tracewin-only requires --tracewin")

    if args.quick:
        args.eval_budget     = 30
        args.svg_episodes    = 1
        args.n_runs          = 1
        args.policy_episodes = 2
        args.svg_baseline    = True

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Surrogate: {args.surrogate}")
    surrogate = ModularMLP.load(args.surrogate)
    surrogate.eval()

    print(f"Dataset:   {args.dataset}")
    dataset = BeamDataset.load(args.dataset)

    results: Dict = {}

    if not args.policy_only:
        for run in range(args.n_runs):
            seed = 42 + run
            print(f"\n{'='*65}\nRun {run+1}/{args.n_runs}  (seed={seed})\n{'='*65}")

            print("Bayesian Optimization...")
            r = run_bo(surrogate, dataset, args.eval_budget, seed)
            results.setdefault("bayesian_opt", []).append(r)
            print(f"  best={r['best_score']:.3f}")

            if args.svg_baseline:
                for name, weights in STAGE_WEIGHT_CONFIGS.items():
                    label = f"svg_{name}"
                    print(f"SVGAgent [{name}]...")
                    r = run_svg(surrogate, dataset, args.svg_episodes, args.svg_horizon,
                                seed, weights)
                    results.setdefault(label, []).append(r)
                    print(f"  best={r['best_score']:.3f}")

        print_table(results)

    policy_evaluation = {}
    if not args.tracewin_only:
        policy_evaluation = run_policy_benchmark(
            args, surrogate, dataset, optimization_results=results,
        )

    policy_evaluation_tracewin = {}
    if args.tracewin:
        checkpoint_values = [getattr(args, algo, None) for algo in POLICY_ALGORITHMS]
        if args.sb3_sac is not None and args.sac is None:
            checkpoint_values.append(args.sb3_sac)
        n_ckpts = sum(path is not None for path in checkpoint_values)
        est_min = n_ckpts * args.tracewin_episodes * args.max_ep_steps * 30 / 60
        print(f"\nTraceWin validation: {n_ckpts} policies × {args.tracewin_episodes} "
              f"episodes × {args.max_ep_steps} steps ≈ {est_min:.0f} min of real physics")

        def tracewin_env_factory():
            from beam_optimization.env.tracewin_env import TraceWinEnv
            return TraceWinEnv(
                project_file=args.tracewin, max_steps=args.max_ep_steps,
                reset_scale=TEST_RESET_SCALE,
                kill_stale=not args.no_kill_stale,
                num_threads=args.tracewin_threads,
                timeout=args.tracewin_timeout,
            )

        policy_evaluation_tracewin = run_policy_benchmark(
            args, surrogate, dataset,
            env_factory=tracewin_env_factory,
            episodes=args.tracewin_episodes,
            tag="_tracewin",
            optimization_results=results,
        )

    output_payload = {"score_function": score_function_metadata()}
    if not args.policy_only:
        output_payload["optimization_results"] = results
    if not args.tracewin_only:
        output_payload["policy_evaluation"] = policy_evaluation
    if args.tracewin:
        output_payload["policy_evaluation_tracewin"] = policy_evaluation_tracewin

    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nResults saved → {args.output}")

    if results:
        try:
            plot_path = save_summary_plot(results, args.output)
            print(f"Plot saved   → {plot_path}")
            convergence_path = save_convergence_plot(results, args.output, args.svg_horizon)
            if convergence_path is not None:
                print(f"Convergence plot saved → {convergence_path}")
        except Exception as exc:
            print(f"WARN: could not save the benchmark plots: {exc}")


if __name__ == "__main__":
    main()
