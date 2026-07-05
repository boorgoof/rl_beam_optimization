"""
Benchmark — compares optimization methods on the surrogate.

All methods receive the same surrogate, the same initial beam0 and the same
evaluation budget.

Methods:
  pso               Particle Swarm Optimization
  bayesian_opt      Bayesian Optimization (GP)
  svg_finale        SVGAgent — final-stage reward only
  svg_uniform      SVGAgent — uniform reward over all stages
  trained policies  Any checkpoint passed via --sac/--td3/.../--svg-uniform
                    is evaluated over --policy-episodes independent episodes.

Usage:
    python -m beam_optimization benchmark \\
        --surrogate env/surrogate_env/surrogate/trained_models/base/surrogate_0.pt \\
        --dataset   env/dataset/base/dataset_base.pt \\
        --output    results/benchmark.json \\
        --n-runs    3 \\
        --eval-budget 3000 \\
        --svg-episodes 500

With trained agents:
    python -m beam_optimization benchmark \\
        --sac runs/all/sac/sac_agent.pt \\
        --td3 runs/all/td3/td3_agent.pt \\
        --ppo runs/all/ppo/ppo_agent.pt

Quick smoke test:
    python -m beam_optimization benchmark --quick
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from beam_optimization.algorithms import make_agent
from beam_optimization.config.paths import (
    DEFAULT_BASE_DATASET,
    DEFAULT_BENCHMARK_OUTPUT,
    DEFAULT_SINGLE_SURROGATE_MODEL,
    DEFAULT_TRACEWIN_INI,
    configure_matplotlib_cache,
)
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.scripts.common import algo_style, run_episode
from beam_optimization.config.adige import (
    MAX_STEPS,
    N_PARAMS,
    PARAM_KEYS,
    action_bounds,
    default_params,
    observation_dim,
    params_to_stage_tensors,
    sensitivity_vec,
    BEAM_STATE_FEATURES,
    score,
)

OBS_DIM = observation_dim()
ACT_DIM = N_PARAMS                               # 16
DEFAULT_PARAM_VALUES = tuple(default_params()[key] for key in PARAM_KEYS)
SENSITIVITY_VALUES = tuple(float(v) for v in sensitivity_vec())
ACTION_BOUNDS = tuple(v.tolist() for v in action_bounds())

STAGE_WEIGHT_CONFIGS = {
    "finale":  None,
    "uniform": [1.0] * 11,
}


# ── Benchmark functions ───────────────────────────────────────────────────────

def run_pso(surrogate, dataset, budget, seed) -> Dict:
    from beam_optimization.algorithms.baselines.pso import PSOOptimizer
    beam0 = _pick_beam(dataset, seed)
    surrogate.eval()

    def objective(params):
        with torch.no_grad():
            outs = surrogate(params_to_stage_tensors(params), beam0)
            return score({v: float(outs[-1][0, i]) for i, v in enumerate(BEAM_STATE_FEATURES)})

    n_particles  = 30
    n_iterations = max(1, budget // n_particles - 1)
    result = PSOOptimizer(
        n_particles=n_particles,
        n_iterations=n_iterations,
        seed=seed,
        param_keys=PARAM_KEYS,
        default_values=DEFAULT_PARAM_VALUES,
        sensitivity_values=SENSITIVITY_VALUES,
    ).optimize(objective)
    return {"best_score": result.best_score, "history": result.score_history}


def run_bo(surrogate, dataset, budget, seed) -> Dict:
    from beam_optimization.algorithms.baselines.bayesian_opt import BayesianOptimizer
    beam0 = _pick_beam(dataset, seed)
    surrogate.eval()

    def objective(params):
        with torch.no_grad():
            outs = surrogate(params_to_stage_tensors(params), beam0)
            return score({v: float(outs[-1][0, i]) for i, v in enumerate(BEAM_STATE_FEATURES)})

    result = BayesianOptimizer(
        n_calls=min(budget, 200),
        seed=seed,
        param_keys=PARAM_KEYS,
        default_values=DEFAULT_PARAM_VALUES,
        sensitivity_values=SENSITIVITY_VALUES,
    ).optimize(objective)
    return {"best_score": result.best_score, "history": result.score_history}


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

    if algo == "sb3_sac":
        from beam_optimization.algorithms.model_free.sb3_sac import SB3SAC
        return SB3SAC.load(ckpt_path, env=env)
    if algo in {"svg_finale", "svg_uniform"}:
        from beam_optimization.algorithms.model_based.svg import SVGAgent
        stage_weights = STAGE_WEIGHT_CONFIGS["uniform" if algo == "svg_uniform" else "finale"]
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
        name = "sac" if algo == "mbpo" else algo
        agent = make_agent(name, obs_dim, ACT_DIM, bounds, hidden_dims=hidden)

    agent.load(ckpt_path)
    return agent


def run_policy_episode(env, agent, algo: str, seed: int, episode_idx: int) -> dict:
    result = run_episode(env, agent, seed=seed)
    features = result["final_features"]
    final_ex = float(features.get("ex", np.nan))
    final_ey = float(features.get("ey", np.nan))
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
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for algo, row in sorted(summary.items()):
            writer.writerow({"algorithm": algo, **row})

    return episodes_path, summary_path


# Panels of the policy benchmark figures: (metric key, title, y-label).
POLICY_PANELS = [
    ("final_score", "Final score", "score (higher is better)"),
    ("final_emittance", "Final emittance", "(ex + ey) / 2 (lower is better)"),
    ("final_npart_ratio", "Final particle ratio", "npart ratio (higher is better)"),
    ("total_reward", "Score improvement", "Σ reward = final − initial score"),
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
            for ref in ("bayesian_opt", "pso"):
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
            model=surrogate, dataset=dataset, max_steps=args.max_ep_steps
        )
    if episodes is None:
        episodes = args.policy_episodes

    checkpoint_args = {
        "sac": args.sac,
        "td3": args.td3,
        "ppo": args.ppo,
        "ddpg": args.ddpg,
        "a2c": args.a2c,
        "reinforce": args.reinforce,
        "trpo": args.trpo,
        "sb3_sac": args.sb3_sac,
        "mbpo": args.mbpo,
        "svg_finale": args.svg_finale,
        "svg_uniform": args.svg_uniform,
    }
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
    print_policy_table(summary)
    episodes_csv, summary_csv = write_policy_csvs(all_episodes, summary, args.output, tag=tag)
    print(f"Policy episode CSV saved → {episodes_csv}")
    print(f"Policy summary CSV saved → {summary_csv}")

    plot_paths = {}
    if not args.no_policy_plots:
        bar_path, box_path = save_policy_plots(
            all_episodes, summary, args.output,
            tag=tag, optimization_results=optimization_results,
        )
        plot_paths = {"bar_plot": str(bar_path), "box_plot": str(box_path)}
        print(f"Policy bar plot saved → {bar_path}")
        print(f"Policy boxplot saved  → {box_path}")

    return {
        "episodes": all_episodes,
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

    PSO/BO histories contain one entry per objective evaluation; SVG histories
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
    parser.add_argument("--surrogate",    default=str(DEFAULT_SINGLE_SURROGATE_MODEL))
    parser.add_argument("--dataset",      default=str(DEFAULT_BASE_DATASET))
    parser.add_argument("--output",       default=str(DEFAULT_BENCHMARK_OUTPUT))
    parser.add_argument("--n-runs",       type=int, default=3)
    parser.add_argument("--eval-budget",  type=int, default=3000)
    parser.add_argument("--svg-episodes", type=int, default=500)
    parser.add_argument("--svg-horizon",  type=int, default=MAX_STEPS)
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
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer sizes used to recreate checkpointed custom agents.")
    parser.add_argument("--no-policy-plots", action="store_true",
                        help="Disable policy bar plot and boxplot generation.")
    parser.add_argument("--quick",        action="store_true")
    parser.add_argument("--sac",          default=None, metavar="CKPT")
    parser.add_argument("--td3",          default=None, metavar="CKPT")
    parser.add_argument("--ppo",          default=None, metavar="CKPT")
    parser.add_argument("--ddpg",         default=None, metavar="CKPT")
    parser.add_argument("--a2c",          default=None, metavar="CKPT")
    parser.add_argument("--reinforce",    default=None, metavar="CKPT")
    parser.add_argument("--trpo",         default=None, metavar="CKPT")
    parser.add_argument("--sb3-sac",      dest="sb3_sac", default=None, metavar="CKPT")
    parser.add_argument("--mbpo",         default=None, metavar="CKPT")
    parser.add_argument("--svg-finale",   dest="svg_finale", default=None, metavar="CKPT")
    parser.add_argument("--svg-uniform",  dest="svg_uniform", default=None, metavar="CKPT")
    args = parser.parse_args()

    if args.quick:
        args.eval_budget     = 30
        args.svg_episodes    = 1
        args.n_runs          = 1
        args.policy_episodes = 2

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Surrogate: {args.surrogate}")
    surrogate = ModularMLP.load(args.surrogate)
    surrogate.eval()

    print(f"Dataset:   {args.dataset}")
    dataset = BeamDataset.load(args.dataset)

    results: Dict = {}

    for run in range(args.n_runs):
        seed = 42 + run
        print(f"\n{'='*65}\nRun {run+1}/{args.n_runs}  (seed={seed})\n{'='*65}")

        print("PSO...")
        r = run_pso(surrogate, dataset, args.eval_budget, seed)
        results.setdefault("pso", []).append(r)
        print(f"  best={r['best_score']:.3f}")

        print("Bayesian Optimization...")
        r = run_bo(surrogate, dataset, args.eval_budget, seed)
        results.setdefault("bayesian_opt", []).append(r)
        print(f"  best={r['best_score']:.3f}")

        for name, weights in STAGE_WEIGHT_CONFIGS.items():
            label = f"svg_{name}"
            print(f"SVGAgent [{name}]...")
            r = run_svg(surrogate, dataset, args.svg_episodes, args.svg_horizon,
                        seed, weights)
            results.setdefault(label, []).append(r)
            print(f"  best={r['best_score']:.3f}")

    print_table(results)

    policy_evaluation = run_policy_benchmark(
        args, surrogate, dataset, optimization_results=results,
    )

    policy_evaluation_tracewin = {}
    if args.tracewin:
        n_ckpts = sum(
            1 for path in (args.sac, args.td3, args.ppo, args.ddpg, args.a2c,
                           args.reinforce, args.trpo, args.sb3_sac, args.mbpo,
                           args.svg_finale, args.svg_uniform)
            if path is not None
        )
        est_min = n_ckpts * args.tracewin_episodes * args.max_ep_steps * 30 / 60
        print(f"\nTraceWin validation: {n_ckpts} policies × {args.tracewin_episodes} "
              f"episodes × {args.max_ep_steps} steps ≈ {est_min:.0f} min of real physics")

        def tracewin_env_factory():
            from beam_optimization.env.tracewin_env import TraceWinEnv
            return TraceWinEnv(project_file=args.tracewin, max_steps=args.max_ep_steps)

        policy_evaluation_tracewin = run_policy_benchmark(
            args, surrogate, dataset,
            env_factory=tracewin_env_factory,
            episodes=args.tracewin_episodes,
            tag="_tracewin",
            optimization_results=results,
        )

    output_payload = {
        "optimization_results": results,
        "policy_evaluation": policy_evaluation,
        "policy_evaluation_tracewin": policy_evaluation_tracewin,
    }

    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nResults saved → {args.output}")

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
