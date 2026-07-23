"""
Train — trains all algorithms in sequence on the surrogate environment.

Algorithms:
  model-free      SAC, TD3, PPO, DDPG, A2C, REINFORCE, TRPO (+ SB3-SAC baseline)
  model-based     SVGAgent, MBPO (with inner SAC)

Quick smoke test (few steps):
    python -m beam_optimization train_policies --quick

Full run:
    python -m beam_optimization train_policies \\
        --dataset   env/dataset/001/dataset_all.pt \\
        --single-surrogate env/surrogate_env/surrogate/trained_models/base/surrogate_001_0.pt \\
        --base-ensemble env/surrogate_env/surrogate/trained_models/base \\
        --output    results/train/rl/all \\
        --rl-steps  300000 \\
        --svg-episodes 2000

Checkpoints are saved as:
    results/train/rl/all/<algo>/<algo>_agent.pt          (model-free)
    results/train/rl/all/sb3_sac/sb3_sac_agent.zip
    results/train/rl/all/svg_finale/svg_agent.pt, results/train/rl/all/svg_uniform/svg_agent.pt
    results/train/rl/all/dyna/dyna_agent.pt              (MBPO inner SAC)
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from beam_optimization.algorithms import MODEL_FREE_ALGORITHMS, is_on_policy, make_agent
from beam_optimization.algorithms.utils.logger import Logger
from beam_optimization.config.paths import (
    DEFAULT_BASE_SURROGATE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRACEWIN_INI,
    DEFAULT_UPDATED_SURROGATE_DIR,
    configure_matplotlib_cache,
    default_dataset_path,
    default_single_surrogate_model,
)
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.scripts.common import algo_style, evaluate_policy, set_global_seed
from beam_optimization.config.adige import (
    MAX_STEPS,
    N_OUTPUT_STAGES,
    N_PARAMS,
    PARAM_KEYS,
    TEST_RESET_SCALE,
    TRAIN_RECOVERY_RESET_PROBABILITY,
    TRAIN_RESET_SCALE,
    action_bounds,
    default_params,
    observation_dim,
)

ACT_DIM = N_PARAMS

# Fixed seed for periodic policy evaluations: every algorithm and every
# training seed is evaluated on the same initial states, so learning curves
# are reproducible and comparisons are paired.
EVAL_SEED = 10_000
# OBS_DIM is computed dynamically from the env observation mask.

STAGE_WEIGHT_CONFIGS = {
    "finale":  None,
    "uniform": [1.0] * N_OUTPUT_STAGES,
}


# ── Surrogate loading ─────────────────────────────────────────────────────────

def _surrogate_files(folder: Path) -> List[Path]:
    return sorted(folder.glob("surrogate_*.pt"))


def _missing_surrogate_message(path: Path) -> str:
    return (
        f"Surrogate checkpoint not found: {path}\n"
        "Create base surrogates with:\n"
        "  python -m beam_optimization build_dataset --target-samples N\n"
        "  python -m beam_optimization train_surrogate --train-dataset <dataset_train.pt>"
    )


def load_single_surrogate(path: str | Path):
    """Load the single base surrogate used by model-free algorithms."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_missing_surrogate_message(path))
    model = ModularMLP.load(str(path))
    model.eval()
    print(f"Loaded single surrogate for model-free algorithms: {path}")
    return model


def load_surrogate_ensemble(folder: str | Path, *, label: str):
    """Load all surrogate_*.pt files from a folder in deterministic order."""
    folder = Path(folder)
    model_files = _surrogate_files(folder)
    if not model_files:
        raise FileNotFoundError(
            f"No surrogate_*.pt found in {folder}\n"
            "Create base surrogates with:\n"
            "  python -m beam_optimization build_dataset --target-samples N\n"
        "  python -m beam_optimization train_surrogate --train-dataset <dataset_train.pt>"
        )
    ensemble = [ModularMLP.load(str(path)) for path in model_files]
    for model in ensemble:
        model.eval()
    print(f"Loaded {len(ensemble)} surrogates for {label}: {folder}")
    return ensemble


def initialize_updated_ensemble_from_base(base_dir: str | Path, updated_dir: str | Path) -> None:
    """Copy base surrogate_*.pt files into updated when updated is empty."""
    base_dir = Path(base_dir)
    updated_dir = Path(updated_dir)

    if _surrogate_files(updated_dir):
        return

    base_files = _surrogate_files(base_dir)
    if not base_files:
        raise FileNotFoundError(
            f"Cannot initialize {updated_dir}: no surrogate_*.pt files in {base_dir}"
        )

    updated_dir.mkdir(parents=True, exist_ok=True)
    for source in base_files:
        shutil.copy2(source, updated_dir / source.name)
    print(
        f"Initialized updated ensemble by copying {len(base_files)} surrogates "
        f"from {base_dir} to {updated_dir}"
    )


def _make_logger(out_dir: Path, algorithm: str, enable_tensorboard: bool):
    if not enable_tensorboard:
        return None
    return Logger(out_dir, algorithm=algorithm)


def _loss_metrics(algo: str, losses) -> dict[str, float]:
    if losses is None:
        return {}
    values = list(losses) if isinstance(losses, (tuple, list)) else [losses]
    values = [None if value is None else float(value) for value in values]

    if algo == "sac":
        names = ["critic_loss", "actor_loss", "entropy_loss"]
    elif algo in {"td3", "ddpg"}:
        names = ["critic_loss", "actor_loss", "entropy_loss"]
    else:
        names = ["value_loss", "policy_loss"]

    return {
        name: value
        for name, value in zip(names, values)
        if value is not None
    }


class LearningCurveRecorder:
    """Collect periodic policy evaluations and save CSV/PNG learning curves."""

    fieldnames = [
        "step",
        "episode",
        "eval_mean_reward",
        "eval_mean_score",
        "eval_std_score",
        "eval_best_score",
        "eval_episodes",
    ]

    def __init__(self, out_dir: Path, algorithm: str):
        self.out_dir = Path(out_dir)
        self.algorithm = algorithm
        self.rows: list[dict[str, float]] = []

    def add(self, *, step: int, episode: int, metrics: dict[str, float]) -> None:
        row = {
            "step": int(step),
            "episode": int(episode),
            "eval_mean_reward": float(metrics["mean_reward"]),
            "eval_mean_score": float(metrics["mean_score"]),
            "eval_std_score": float(metrics["std_score"]),
            "eval_best_score": float(metrics["best_score"]),
            "eval_episodes": int(metrics["episodes"]),
        }
        self.rows.append(row)
        self.save_csv()
        self.save_plot()

    def save_csv(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "learning_curve.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def save_plot(self) -> Optional[Path]:
        if not self.rows:
            return None
        configure_matplotlib_cache()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [row["step"] for row in self.rows]
        scores = [row["eval_mean_score"] for row in self.rows]
        stds = [row["eval_std_score"] for row in self.rows]

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(steps, scores, marker="o", linewidth=1.8, label=self.algorithm)
        if len(steps) > 1:
            lower = np.asarray(scores) - np.asarray(stds)
            upper = np.asarray(scores) + np.asarray(stds)
            ax.fill_between(steps, lower, upper, alpha=0.18)
        ax.set_title(f"{self.algorithm} learning curve")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Mean evaluation score")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        path = self.out_dir / "learning_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path


def log_learning_curve_eval(
    *,
    recorder: Optional[LearningCurveRecorder],
    logger,
    step: int,
    episode: int,
    metrics: dict[str, float],
) -> None:
    if recorder is not None:
        recorder.add(step=step, episode=episode, metrics=metrics)
    if logger is not None:
        logger.log(
            {
                "eval/mean_reward": float(metrics["mean_reward"]),
                "eval/mean_score": float(metrics["mean_score"]),
                "eval/std_score": float(metrics["std_score"]),
                "eval/best_score": float(metrics["best_score"]),
                "eval/episodes": float(metrics["episodes"]),
            },
            step=step,
        )
    print(
        f"  eval step={step:>7d}  mean_reward={metrics['mean_reward']:.2f}  "
        f"mean_score={metrics['mean_score']:.3f}  best_eval={metrics['best_score']:.3f}"
    )


def save_all_learning_curves_plot(curves: Dict[str, list[dict]], out_root: Path) -> Optional[Path]:
    curves = {name: rows for name, rows in curves.items() if rows}
    if not curves:
        return None

    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for name, rows in sorted(curves.items()):
        steps = np.asarray([row["step"] for row in rows], dtype=float)
        scores = np.asarray([row["eval_mean_score"] for row in rows], dtype=float)
        stds = np.asarray([row.get("eval_std_score", 0.0) for row in rows], dtype=float)
        color, linestyle = algo_style(name)
        ax.plot(steps, scores, color=color, linestyle=linestyle, linewidth=1.8, label=name)
        if len(steps) > 1:
            ax.fill_between(steps, scores - stds, scores + stds, color=color, alpha=0.12)
    ax.set_title("Learning curves")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean evaluation score (higher is better)")
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()

    path = out_root / "learning_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ── Multi-seed orchestration ──────────────────────────────────────────────────

def aggregate_seed_curves(per_seed_rows: list[list[dict]]) -> list[dict]:
    """Aggregate per-seed learning curves into one mean±std curve.

    Seeds share the same eval-step grid; steps present in every seed are kept.
    The returned std is the across-seed std of the per-seed mean scores.
    """
    per_seed_rows = [rows for rows in per_seed_rows if rows]
    if not per_seed_rows:
        return []
    common_steps = sorted(
        set.intersection(*({row["step"] for row in rows} for rows in per_seed_rows))
    )
    by_step = [
        {row["step"]: row for row in rows}
        for rows in per_seed_rows
    ]
    aggregated = []
    for step in common_steps:
        scores = np.asarray([m[step]["eval_mean_score"] for m in by_step], dtype=float)
        rewards = np.asarray([m[step]["eval_mean_reward"] for m in by_step], dtype=float)
        bests = np.asarray([m[step]["eval_best_score"] for m in by_step], dtype=float)
        aggregated.append({
            "step": int(step),
            "episode": int(max(m[step]["episode"] for m in by_step)),
            "eval_mean_reward": float(rewards.mean()),
            "eval_mean_score": float(scores.mean()),
            "eval_std_score": float(scores.std()),
            "eval_best_score": float(bests.max()),
            "eval_episodes": int(by_step[0][step]["eval_episodes"]),
        })
    return aggregated


def run_seeded(
    label: str,
    out_root: Path,
    seeds: list[int],
    checkpoint_files: list[str],
    train_fn,
    learning_curves: Optional[Dict[str, list[dict]]],
) -> dict:
    """Run train_fn once per seed, aggregate curves, promote the best checkpoint.

    train_fn(seed, out_dir, curves) must train one agent, save its checkpoint(s)
    in out_dir, fill `curves[<any key>]` with learning-curve rows, and return the
    best training score. With one seed everything stays in out_root/label as
    before; with N seeds each run lives in out_root/label/seed_<s> and the best
    seed's checkpoint files are copied up to out_root/label/.
    """
    multi = len(seeds) > 1
    per_seed_scores: list[float] = []
    per_seed_curves: list[list[dict]] = []
    selection_scores: list[float] = []

    for s in seeds:
        if multi:
            print(f"\n--- {label} seed={s} ---")
        out_dir = out_root / label / (f"seed_{s}" if multi else "")
        curves: Dict[str, list[dict]] = {}
        best = train_fn(seed=s, out_dir=out_dir, curves=curves)
        rows = next(iter(curves.values()), [])
        per_seed_scores.append(float(best))
        per_seed_curves.append(rows)
        # Select the best seed by final evaluation score (more robust than the
        # best single training episode); fall back to the training best.
        selection_scores.append(rows[-1]["eval_mean_score"] if rows else float(best))

    aggregated = aggregate_seed_curves(per_seed_curves)
    if learning_curves is not None and aggregated:
        learning_curves[label] = aggregated

    best_idx = int(np.argmax(selection_scores))
    if multi:
        # Aggregated curve + promoted checkpoint at the algo level, so the paths
        # used by benchmark/test do not depend on the number of seeds.
        recorder = LearningCurveRecorder(out_root / label, label)
        recorder.rows = aggregated
        if aggregated:
            recorder.save_csv()
            recorder.save_plot()
        for fname in checkpoint_files:
            src = out_root / label / f"seed_{seeds[best_idx]}" / fname
            if src.exists():
                shutil.copy2(src, out_root / label / fname)
        print(f"  Best seed for {label}: {seeds[best_idx]} "
              f"(final eval {selection_scores[best_idx]:.3f}) → promoted checkpoint")

    return {
        "best_score_mean": float(np.mean(per_seed_scores)),
        "best_score_std": float(np.std(per_seed_scores)),
        "best_seed": int(seeds[best_idx]),
        "per_seed": {str(s): float(v) for s, v in zip(seeds, per_seed_scores)},
    }


# ── Training loops ────────────────────────────────────────────────────────────

def train_rl(algo: str, surrogate, dataset, n_steps, max_ep_steps,
             hidden, out_dir: Path,
             seed: int = 42,
             enable_tensorboard: bool = True,
             eval_every: int = 1000,
             eval_episodes: int = 5,
             enable_learning_curve: bool = True,
             learning_curves: Optional[Dict[str, list[dict]]] = None) -> float:
    """Train one custom model-free algorithm on the surrogate environment."""
    set_global_seed(seed)
    # Create env first so obs_dim is known before building the agent
    env = SurrogateEnv(
        model=surrogate, dataset=dataset, max_steps=max_ep_steps,
        reset_scale=TRAIN_RESET_SCALE,
        recovery_reset_probability=TRAIN_RECOVERY_RESET_PROBABILITY,
    )
    obs_dim = env.observation_space.shape[0]

    act_bds = action_bounds()
    bounds  = (act_bds[0].tolist(), act_bds[1].tolist())
    agent_kwargs = {}
    if not is_on_policy(algo):
        # Scale the replay warmup with the budget so short (--quick) runs
        # still perform gradient updates; full runs keep the default 1000.
        agent_kwargs["warmup_steps"] = min(1000, max(1, n_steps // 4))
    agent = make_agent(algo, obs_dim, ACT_DIM, bounds, hidden_dims=hidden, **agent_kwargs)

    obs, _     = env.reset(seed=seed)
    best_score = -np.inf
    ep_reward  = 0.0
    ep_count   = 0
    log_every  = max(1, n_steps // 20)
    logger = _make_logger(out_dir, algo, enable_tensorboard)
    recorder = (
        LearningCurveRecorder(out_dir, algo)
        if enable_learning_curve and eval_episodes > 0
        else None
    )

    on_policy = is_on_policy(algo)
    make_eval_env = lambda: SurrogateEnv(
        model=surrogate, dataset=dataset, max_steps=max_ep_steps,
        reset_scale=TEST_RESET_SCALE,
    )

    try:
        if recorder is not None:
            metrics = evaluate_policy(agent, make_eval_env, eval_episodes, seed=EVAL_SEED)
            log_learning_curve_eval(
                recorder=recorder,
                logger=logger,
                step=0,
                episode=0,
                metrics=metrics,
            )

        for step in range(1, n_steps + 1):
            if on_policy:
                action, logpa, value = agent.select_action(obs, training=True)
            else:
                action = agent.select_action(obs, training=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store only true terminations: a time-limit truncation must keep
            # its bootstrap (the episode could have continued), otherwise the
            # (1 - done) factor in the TD/GAE targets wrongly zeroes V(s_next)
            # on the last step of every episode.
            if on_policy:
                agent.store(obs, action, reward, value, logpa, float(terminated))
            else:
                agent.store(obs, action, reward, next_obs, float(terminated))

            optimize_result = None
            if on_policy:
                if done:
                    last_val = 0.0
                    if not terminated:
                        # Bootstrap truncated episodes with V(s_{t+1}).
                        _, _, last_val = agent.select_action(next_obs, training=True)
                    optimize_result = agent.optimize(last_value=float(last_val))
            else:
                optimize_result = agent.optimize()

            if logger is not None:
                metrics = {
                    "reward": float(reward),
                    "action_norm": float(np.linalg.norm(action)),
                }
                metrics.update(_loss_metrics(algo, optimize_result))
                if len(metrics) > 2 or step % log_every == 0:
                    logger.log(metrics, step=step)

            obs        = next_obs
            ep_reward += reward

            if done:
                ep_count += 1
                sc = info.get("score", 0.0)
                best_score = max(best_score, sc)
                if logger is not None:
                    logger.log(
                        {
                            "episode_reward": float(ep_reward),
                            "score": float(sc),
                            "best_score": float(best_score),
                            "episode": float(ep_count),
                        },
                        step=step,
                    )
                if step % log_every == 0:
                    print(f"  step={step:>7d}  ep={ep_count:>4d}  "
                          f"reward={ep_reward:.2f}  score={sc:.3f}  best={best_score:.3f}")
                obs, _ = env.reset()
                ep_reward = 0.0

            if recorder is not None and step % max(1, eval_every) == 0:
                metrics = evaluate_policy(agent, make_eval_env, eval_episodes, seed=EVAL_SEED)
                log_learning_curve_eval(
                    recorder=recorder,
                    logger=logger,
                    step=step,
                    episode=ep_count,
                    metrics=metrics,
                )

        if recorder is not None and (not recorder.rows or recorder.rows[-1]["step"] != n_steps):
            metrics = evaluate_policy(agent, make_eval_env, eval_episodes, seed=EVAL_SEED)
            log_learning_curve_eval(
                recorder=recorder,
                logger=logger,
                step=n_steps,
                episode=ep_count,
                metrics=metrics,
            )
        if learning_curves is not None and recorder is not None:
            learning_curves[algo] = list(recorder.rows)

        out_dir.mkdir(parents=True, exist_ok=True)
        agent.save(str(out_dir / f"{algo}_agent.pt"))
        print(f"  Saved → {out_dir / f'{algo}_agent.pt'}")
        return best_score
    finally:
        if logger is not None:
            logger.close()


def train_sb3_sac(surrogate, dataset, n_steps, max_ep_steps,
                  hidden, out_dir: Path,
                  seed: int = 42,
                  enable_tensorboard: bool = True,
                  eval_every: int = 1000,
                  eval_episodes: int = 5,
                  enable_learning_curve: bool = True,
                  learning_curves: Optional[Dict[str, list[dict]]] = None) -> float:
    """Train Stable Baselines 3 SAC on the surrogate environment (sanity baseline)."""
    from beam_optimization.algorithms.model_free.sb3_sac import SB3SAC

    set_global_seed(seed)
    env = SurrogateEnv(
        model=surrogate, dataset=dataset, max_steps=max_ep_steps,
        reset_scale=TRAIN_RESET_SCALE,
        recovery_reset_probability=TRAIN_RECOVERY_RESET_PROBABILITY,
    )

    logger = _make_logger(out_dir, "sb3_sac", enable_tensorboard)
    recorder = (
        LearningCurveRecorder(out_dir, "sb3_sac")
        if enable_learning_curve and eval_episodes > 0
        else None
    )
    make_eval_env = lambda: SurrogateEnv(
        model=surrogate, dataset=dataset, max_steps=max_ep_steps,
        reset_scale=TEST_RESET_SCALE,
    )
    agent = SB3SAC(
        env,
        hidden_dims=tuple(hidden),
        seed=seed,
        tensorboard_log=str(out_dir) if enable_tensorboard else None,
    )
    try:
        best_score = agent.train(
            env,
            n_steps=n_steps,
            log_every=max(1, n_steps // 20),
            logger=logger,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_fn=(lambda current_agent, n: evaluate_policy(current_agent, make_eval_env, n, seed=EVAL_SEED))
            if recorder is not None
            else None,
            eval_logger=(
                lambda step, metrics: log_learning_curve_eval(
                    recorder=recorder,
                    logger=logger,
                    step=step,
                    episode=0,
                    metrics=metrics,
                )
            )
            if recorder is not None
            else None,
        )
        if recorder is not None and (not recorder.rows or recorder.rows[-1]["step"] != n_steps):
            metrics = evaluate_policy(agent, make_eval_env, eval_episodes, seed=EVAL_SEED)
            log_learning_curve_eval(
                recorder=recorder,
                logger=logger,
                step=n_steps,
                episode=0,
                metrics=metrics,
            )
        if learning_curves is not None and recorder is not None:
            learning_curves["sb3_sac"] = list(recorder.rows)
    finally:
        if logger is not None:
            logger.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_dir / "sb3_sac_agent"))
    print(f"  Saved → {out_dir / 'sb3_sac_agent.zip'}")
    return best_score


def train_svg(surrogate, dataset, n_episodes, horizon, hidden,
              stage_weights, out_dir: Path,
              seed: int = 42,
              enable_tensorboard: bool = True,
              eval_every: int = 1000,
              eval_episodes: int = 5,
              enable_learning_curve: bool = True,
              learning_curves: Optional[Dict[str, list[dict]]] = None,
              curve_label: str = "svg") -> float:
    from beam_optimization.algorithms.model_based.svg import SVGAgent

    set_global_seed(seed)
    agent = SVGAgent(
        surrogate=surrogate,
        dataset=dataset,
        obs_dim=observation_dim(),
        act_dim=ACT_DIM,
        action_bounds=tuple(v.tolist() for v in action_bounds()),
        param_keys=PARAM_KEYS,
        default_params=default_params(),
        hidden_dims=tuple(hidden),
        n_step=horizon,
        stage_weights=stage_weights,
    )

    best_score = -np.inf
    log_every  = max(1, n_episodes // 20)
    logger = _make_logger(out_dir, "svg", enable_tensorboard)
    recorder = (
        LearningCurveRecorder(out_dir, curve_label)
        if enable_learning_curve and eval_episodes > 0
        else None
    )
    eval_every_episodes = max(1, max(1, eval_every) // max(1, horizon))
    make_eval_env = lambda: SurrogateEnv(
        model=surrogate, dataset=dataset, max_steps=horizon,
        reset_scale=TEST_RESET_SCALE,
    )

    try:
        if recorder is not None:
            metrics = evaluate_policy(agent, make_eval_env, eval_episodes, seed=EVAL_SEED)
            log_learning_curve_eval(
                recorder=recorder,
                logger=logger,
                step=0,
                episode=0,
                metrics=metrics,
            )

        for ep in range(1, n_episodes + 1):
            result     = agent.optimize_episode()
            best_score = max(best_score, result.final_score)
            if logger is not None:
                logger.log(
                    {
                        "episode_loss": float(result.episode_loss),
                        "final_score": float(result.final_score),
                        "best_score": float(best_score),
                        "grad_norm": float(result.grad_norm),
                        "episode": float(ep),
                    },
                    step=ep,
                )
            if ep % log_every == 0:
                print(f"  ep={ep:>5d}  loss={result.episode_loss:.4f}  "
                      f"score={result.final_score:.3f}  best={best_score:.3f}")
            if recorder is not None and ep % eval_every_episodes == 0:
                step = ep * horizon
                metrics = evaluate_policy(agent, make_eval_env, eval_episodes, seed=EVAL_SEED)
                log_learning_curve_eval(
                    recorder=recorder,
                    logger=logger,
                    step=step,
                    episode=ep,
                    metrics=metrics,
                )

        final_step = n_episodes * horizon
        if recorder is not None and (not recorder.rows or recorder.rows[-1]["step"] != final_step):
            metrics = evaluate_policy(agent, make_eval_env, eval_episodes, seed=EVAL_SEED)
            log_learning_curve_eval(
                recorder=recorder,
                logger=logger,
                step=final_step,
                episode=n_episodes,
                metrics=metrics,
            )
        if learning_curves is not None and recorder is not None:
            learning_curves[curve_label] = list(recorder.rows)

        out_dir.mkdir(parents=True, exist_ok=True)
        agent.save(str(out_dir / "svg_agent.pt"))
        print(f"  Saved → {out_dir / 'svg_agent.pt'}")
        return best_score
    finally:
        if logger is not None:
            logger.close()


def train_dyna(surrogate, dataset, n_steps, max_ep_steps,
               rollout_length, hidden, out_dir: Path,
               seed: int = 42,
               tracewin_project: Optional[str] = None,
               online_finetune: bool = False,
               online_mix_ratio: float = 0.5,
               update_dataset_path: Optional[str] = None,
               surrogate_path: Optional[Path] = None,
               dataset_path: Optional[Path] = None,
               update_surrogates_path: Optional[str] = None,
               enable_tensorboard: bool = True,
               eval_every: int = 1000,
               eval_episodes: int = 5,
               enable_learning_curve: bool = True,
               learning_curves: Optional[Dict[str, list[dict]]] = None) -> float:
    """
    Train MBPO with the surrogate ensemble for synthetic rollouts.

    If tracewin_project is given, the REAL environment uses TraceWin (true
    physics, ~30 s/step). SYNTHETIC rollouts always use the surrogate (fast).
    Without tracewin_project both the real env and the rollouts use the
    surrogate.

    If online_finetune=True (requires tracewin_project), MBPOWithModelUpdate
    fine-tunes the surrogate every model_train_freq real steps on a mix of
    offline (original dataset) and online (TraceWin data from this run,
    fraction online_mix_ratio) samples. In that case:
      - fine-tuned weights go to models/updated: starting from models/base
        preserves base; starting from models/updated updates the working copy
        in-place. Override with update_surrogates_path;
      - the merged offline+online dataset is saved by default to the same file
        passed with --dataset (whatever that resolves to, so the offline
        dataset grows run after run). Use update_dataset_path to save
        elsewhere.
    """
    from beam_optimization.algorithms.model_free.sac import SAC

    set_global_seed(seed)
    use_model_update = online_finetune and bool(tracewin_project)

    if use_model_update:
        from beam_optimization.algorithms.model_based.mbpo_model_update import MBPOWithModelUpdate
        MBPOClass = MBPOWithModelUpdate
    else:
        from beam_optimization.algorithms.model_based.mbpo import MBPO
        MBPOClass = MBPO

    if tracewin_project:
        from beam_optimization.env.tracewin_env import TraceWinEnv
        env = TraceWinEnv(
            project_file=tracewin_project,
            max_steps=max_ep_steps,
            reset_scale=TRAIN_RESET_SCALE,
            recovery_reset_probability=TRAIN_RECOVERY_RESET_PROBABILITY,
        )
        label = "MBPOWithModelUpdate" if use_model_update else "MBPO"
        print(f"  Real env: TraceWin  ({tracewin_project})  [{label}]")
    else:
        env = SurrogateEnv(
            model=surrogate, dataset=dataset, max_steps=max_ep_steps,
            reset_scale=TRAIN_RESET_SCALE,
            recovery_reset_probability=TRAIN_RECOVERY_RESET_PROBABILITY,
        )
        print("  Real env: surrogate (SurrogateEnv)  [MBPO]")

    obs_dim = env.observation_space.shape[0]
    act_bds = action_bounds()
    bounds  = (act_bds[0].tolist(), act_bds[1].tolist())
    # Scale warmup/thresholds with the budget so short (--quick) runs still train.
    warmup = min(1000, max(1, n_steps // 4))
    inner  = SAC(obs_dim, ACT_DIM, bounds, hidden_dims=tuple(hidden), warmup_steps=warmup)

    mbpo_kwargs = dict(
        agent=inner,
        surrogates=surrogate,
        dataset=dataset,
        obs_dim=obs_dim,
        act_dim=ACT_DIM,
        rollout_length=rollout_length,
        min_real_samples=min(256, max(1, n_steps // 4)),
    )
    if use_model_update:
        mbpo_kwargs["online_mix_ratio"] = online_mix_ratio

        if update_dataset_path is not None:
            dataset_save_path = Path(update_dataset_path)
        elif dataset_path is not None:
            # By default MBPOWithModelUpdate writes the merged offline+online
            # dataset back to the base dataset used for beam0/offline samples.
            dataset_save_path = dataset_path
        else:
            dataset_save_path = out_dir / "updated_dataset.pt"
        mbpo_kwargs["dataset_save_path"] = dataset_save_path

        if update_surrogates_path is not None:
            surrogate_save_dir = Path(update_surrogates_path)
        elif surrogate_path is not None:
            base_dir = surrogate_path if surrogate_path.is_dir() else surrogate_path.parent
            if base_dir.name == "base":
                surrogate_save_dir = base_dir.parent / "updated"
            elif base_dir.name == "updated":
                surrogate_save_dir = base_dir
            else:
                surrogate_save_dir = base_dir / "updated"
        else:
            surrogate_save_dir = out_dir / "updated"
        mbpo_kwargs["surrogate_save_dir"] = surrogate_save_dir
    mbpo = MBPOClass(**mbpo_kwargs)
    # Seed the synthetic-rollout env once; later resets reuse this generator.
    mbpo.synthetic_env.reset(seed=seed)

    obs, _     = env.reset(seed=seed)
    best_score = -np.inf
    ep_reward  = 0.0
    ep_count   = 0
    log_every  = max(1, n_steps // 20)
    logger = _make_logger(out_dir, "mbpo", enable_tensorboard)
    recorder = (
        LearningCurveRecorder(out_dir, "mbpo")
        if enable_learning_curve and eval_episodes > 0
        else None
    )

    def make_eval_env():
        if tracewin_project:
            from beam_optimization.env.tracewin_env import TraceWinEnv
            return TraceWinEnv(
                project_file=tracewin_project, max_steps=max_ep_steps,
                reset_scale=TEST_RESET_SCALE,
            )
        return SurrogateEnv(
            model=surrogate, dataset=dataset, max_steps=max_ep_steps,
            reset_scale=TEST_RESET_SCALE,
        )

    try:
        if recorder is not None:
            metrics = evaluate_policy(mbpo, make_eval_env, eval_episodes, seed=EVAL_SEED)
            log_learning_curve_eval(
                recorder=recorder,
                logger=logger,
                step=0,
                episode=0,
                metrics=metrics,
            )

        for step in range(1, n_steps + 1):
            action   = mbpo.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done     = terminated or truncated

            # Pass only true terminations to the replay buffer (see train_rl):
            # `done` keeps driving the episode loop below.
            if use_model_update:
                optimize_result = mbpo.step(obs, action, reward, next_obs, terminated,
                                            sim_result=info.get("sim_result"))
            else:
                optimize_result = mbpo.step(obs, action, reward, next_obs, terminated)

            if logger is not None:
                metrics = {
                    "real_reward": float(reward),
                    "reward": float(reward),
                    "action_norm": float(np.linalg.norm(action)),
                }
                metrics.update(_loss_metrics("sac", optimize_result))
                if use_model_update:
                    metrics["online_samples"] = float(mbpo.n_online_samples)
                    update_losses = getattr(mbpo, "last_update_losses", None)
                    update_freq = max(1, int(getattr(mbpo, "model_train_freq", 1)))
                    if update_losses is not None and step % update_freq == 0:
                        update_values = [float(v) for v in update_losses.values()]
                        if update_values:
                            metrics["model_update_loss"] = float(np.mean(update_values))
                        for name, value in update_losses.items():
                            metrics[f"model_update_loss/{name}"] = float(value)
                if len(metrics) > 3 or step % log_every == 0:
                    logger.log(metrics, step=step)

            obs       = next_obs
            ep_reward += reward

            if done:
                ep_count  += 1
                sc         = info.get("score", 0.0)
                best_score = max(best_score, sc)
                if logger is not None:
                    metrics = {
                        "episode_reward": float(ep_reward),
                        "score": float(sc),
                        "best_score": float(best_score),
                        "episode": float(ep_count),
                    }
                    if use_model_update:
                        metrics["online_samples"] = float(mbpo.n_online_samples)
                    logger.log(metrics, step=step)
                if step % log_every == 0:
                    extra = (f"  online_samples={mbpo.n_online_samples}"
                             if use_model_update else "")
                    print(f"  step={step:>7d}  ep={ep_count:>4d}  "
                          f"reward={ep_reward:.2f}  score={sc:.3f}  best={best_score:.3f}{extra}")
                obs, _ = env.reset()
                ep_reward = 0.0

            if recorder is not None and step % max(1, eval_every) == 0:
                metrics = evaluate_policy(mbpo, make_eval_env, eval_episodes, seed=EVAL_SEED)
                log_learning_curve_eval(
                    recorder=recorder,
                    logger=logger,
                    step=step,
                    episode=ep_count,
                    metrics=metrics,
                )

        if recorder is not None and (not recorder.rows or recorder.rows[-1]["step"] != n_steps):
            metrics = evaluate_policy(mbpo, make_eval_env, eval_episodes, seed=EVAL_SEED)
            log_learning_curve_eval(
                recorder=recorder,
                logger=logger,
                step=n_steps,
                episode=ep_count,
                metrics=metrics,
            )
        if learning_curves is not None and recorder is not None:
            learning_curves["mbpo"] = list(recorder.rows)

        out_dir.mkdir(parents=True, exist_ok=True)
        inner.save(str(out_dir / "dyna_agent.pt"))
        print(f"  Saved → {out_dir / 'dyna_agent.pt'}")

        if use_model_update:
            # final flush, even outside the model_train_freq period
            mbpo.save_surrogates()
            mbpo.save_dataset()

        return best_score
    finally:
        if logger is not None:
            logger.close()


# ── Final summary ─────────────────────────────────────────────────────────────

def print_summary(scores: Dict[str, dict]):
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'Algorithm':<25}  {'Best score (mean±std over seeds)':>32}")
    print("-" * 60)
    for name, entry in sorted(scores.items()):
        print(f"{name:<25}  {entry['best_score_mean']:>15.3f} ± {entry['best_score_std']:<8.3f}"
              f" (seed {entry['best_seed']})")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    configure_matplotlib_cache()

    parser = argparse.ArgumentParser(description="Train all algorithms")
    parser.add_argument("--single-surrogate", default=str(default_single_surrogate_model()),
                        metavar="PATH",
                        help="Single surrogate used by SAC/PPO/TD3/DDPG/A2C/"
                             "REINFORCE/TRPO/SB3-SAC. Default: first "
                             "surrogate_*.pt found in --base-ensemble.")
    parser.add_argument("--base-ensemble", default=str(DEFAULT_BASE_SURROGATE_DIR),
                        metavar="PATH",
                        help="Folder with surrogate_*.pt used by SVG and MBPO. "
                             f"Default: {DEFAULT_BASE_SURROGATE_DIR}")
    parser.add_argument("--updated-ensemble", default=str(DEFAULT_UPDATED_SURROGATE_DIR),
                        metavar="PATH",
                        help="Working folder used only by MBPOWithModelUpdate. "
                             "If empty it is initialized by copying --base-ensemble. "
                             f"Default: {DEFAULT_UPDATED_SURROGATE_DIR}")
    parser.add_argument("--dataset",        default=str(default_dataset_path()))
    parser.add_argument("--output",         default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rl-steps",       type=int, default=200_000)
    parser.add_argument("--svg-episodes",   type=int, default=1000)
    parser.add_argument("--svg-horizon",    type=int, default=MAX_STEPS)
    parser.add_argument("--rollout-length", type=int, default=1,
                        help="MBPO synthetic rollout length (1=Dyna, >1=MBPO)")
    parser.add_argument("--max-ep-steps",   type=int, default=MAX_STEPS)
    parser.add_argument("--hidden",         type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed",           type=int, default=42,
                        help="Base random seed. With --n-seeds N, seeds are "
                             "seed, seed+1, ..., seed+N-1. Default: 42.")
    parser.add_argument("--n-seeds",        type=int, default=1,
                        help="Independent training runs per algorithm; learning "
                             "curves become mean±std across seeds and the best "
                             "seed's checkpoint is promoted. Default: 1 "
                             "(thesis runs use 3).")
    parser.add_argument("--quick",          action="store_true",
                        help="Reduced budget for a quick smoke test")
    parser.add_argument("--eval-every",     type=int, default=1000,
                        help="Evaluate the current policy for the learning curve "
                             "every N environment steps. Default: 1000.")
    parser.add_argument("--eval-episodes",  type=int, default=5,
                        help="Test episodes per learning-curve evaluation. "
                             "Default: 5.")
    parser.add_argument("--no-learning-curve", action="store_true",
                        help="Disable periodic evaluation and learning-curve plots.")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard and metrics.csv logging.")
    parser.add_argument("--skip",           nargs="*", default=[],
                        help="Algorithms to skip (e.g.: --skip dyna ppo)")
    parser.add_argument("--tracewin",       default=None, metavar="INI",
                        nargs="?", const=str(DEFAULT_TRACEWIN_INI),
                        help="Use TraceWin as the real env for MBPO. "
                             "Without a value, uses the project default path.")
    parser.add_argument("--online-finetune", action="store_true",
                        help="Fine-tune the surrogate ensemble on real data "
                             "during training (MBPOWithModelUpdate). "
                             "Requires --tracewin.")
    parser.add_argument("--online-mix-ratio", type=float, default=0.5, metavar="FLOAT",
                        help="Target fraction (0-1) of each fine-tuning batch drawn "
                             "from online data collected in this run; the rest comes "
                             "from the original offline dataset (avoids forgetting "
                             "the original distribution). Default: 0.5.")
    parser.add_argument("--update-dataset",  default=None, metavar="PATH",
                        help=".pt path for the merged offline+online dataset collected "
                             "by MBPOWithModelUpdate. Default: same path as --dataset, "
                             "so that file grows run after run. Requires "
                             "--online-finetune.")
    args = parser.parse_args()

    if args.quick:
        args.rl_steps     = 200
        args.svg_episodes = 5

    if args.online_finetune and not args.tracewin:
        parser.error("--online-finetune requires --tracewin")

    out_root = Path(args.output)
    skip     = set(args.skip)
    enable_tensorboard = not args.no_tensorboard
    enable_learning_curve = not args.no_learning_curve

    single_surrogate_path = Path(args.single_surrogate)
    base_ensemble_path = Path(args.base_ensemble)
    updated_ensemble_path = Path(args.updated_ensemble)

    run_model_free = any(algo not in skip for algo in MODEL_FREE_ALGORITHMS)
    run_sb3_sac = "sb3_sac" not in skip
    run_dyna = "dyna" not in skip
    run_svg = "svg" not in skip and any(
        f"svg_{name}" not in skip for name in STAGE_WEIGHT_CONFIGS
    )
    use_model_update = args.online_finetune and bool(args.tracewin)

    single_surrogate = None
    if run_model_free or run_sb3_sac:
        single_surrogate = load_single_surrogate(single_surrogate_path)

    base_ensemble = None
    if run_svg or (run_dyna and not use_model_update):
        base_ensemble = load_surrogate_ensemble(base_ensemble_path, label="base ensemble")

    updated_ensemble = None
    if run_dyna and use_model_update:
        initialize_updated_ensemble_from_base(base_ensemble_path, updated_ensemble_path)
        updated_ensemble = load_surrogate_ensemble(
            updated_ensemble_path,
            label="updated ensemble",
        )

    print(f"Loading dataset:  {args.dataset}")
    dataset = BeamDataset.load(args.dataset)

    scores: Dict[str, float] = {}
    learning_curves: Dict[str, list[dict]] = {}

    seeds = [args.seed + i for i in range(max(1, args.n_seeds))]
    common_kwargs = dict(
        enable_tensorboard=enable_tensorboard,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        enable_learning_curve=enable_learning_curve,
    )

    # ── Model-free RL ─────────────────────────────────────────────────────────
    for algo in MODEL_FREE_ALGORITHMS:
        if algo in skip:
            print(f"\n[SKIP] {algo.upper()}")
            continue
        print(f"\n{'='*50}\nTraining {algo.upper()}  ({args.rl_steps} steps × {len(seeds)} seed)\n{'='*50}")
        scores[algo] = run_seeded(
            algo, out_root, seeds, [f"{algo}_agent.pt"],
            lambda seed, out_dir, curves, algo=algo: train_rl(
                algo, single_surrogate, dataset, args.rl_steps,
                args.max_ep_steps, args.hidden, out_dir,
                seed=seed, learning_curves=curves, **common_kwargs,
            ),
            learning_curves,
        )

    # ── SB3-SAC ───────────────────────────────────────────────────────────────
    if "sb3_sac" not in skip:
        print(f"\n{'='*50}\nTraining SB3-SAC  ({args.rl_steps} steps × {len(seeds)} seed)\n{'='*50}")
        scores["sb3_sac"] = run_seeded(
            "sb3_sac", out_root, seeds, ["sb3_sac_agent.zip"],
            lambda seed, out_dir, curves: train_sb3_sac(
                single_surrogate, dataset, args.rl_steps,
                args.max_ep_steps, args.hidden, out_dir,
                seed=seed, learning_curves=curves, **common_kwargs,
            ),
            learning_curves,
        )

    # ── MBPO ──────────────────────────────────────────────────────────────────
    if "dyna" not in skip:
        tw_label  = "TraceWin" if args.tracewin else "surrogate"
        use_mu    = args.online_finetune and bool(args.tracewin)
        alg_label = "MBPOWithModelUpdate" if use_mu else "MBPO"
        dyna_seeds = seeds
        if args.tracewin and len(seeds) > 1:
            print("WARNING: --tracewin active — MBPO runs with a single seed "
                  "(multi-seed on real physics is prohibitively slow).")
            dyna_seeds = seeds[:1]
        print(f"\n{'='*50}\nTraining {alg_label} [{tw_label}]  ({args.rl_steps} steps × "
              f"{len(dyna_seeds)} seed, rollout={args.rollout_length})\n{'='*50}")
        dyna_surrogate = updated_ensemble if use_mu else base_ensemble
        dyna_surrogate_path = updated_ensemble_path if use_mu else base_ensemble_path
        scores["mbpo"] = run_seeded(
            "dyna", out_root, dyna_seeds, ["dyna_agent.pt"],
            lambda seed, out_dir, curves: train_dyna(
                dyna_surrogate, dataset, args.rl_steps,
                args.max_ep_steps, args.rollout_length, args.hidden, out_dir,
                seed=seed,
                tracewin_project=args.tracewin,
                online_finetune=args.online_finetune,
                online_mix_ratio=args.online_mix_ratio,
                update_dataset_path=args.update_dataset,
                surrogate_path=dyna_surrogate_path,
                dataset_path=Path(args.dataset),
                update_surrogates_path=str(updated_ensemble_path) if use_mu else None,
                learning_curves=curves, **common_kwargs,
            ),
            learning_curves,
        )

    # ── SVGAgent ─────────────────────────────────────────────────────────────
    for name, weights in STAGE_WEIGHT_CONFIGS.items():
        label = f"svg_{name}"
        if label in skip or "svg" in skip:
            print(f"\n[SKIP] {label}")
            continue
        print(f"\n{'='*50}\nTraining SVGAgent [{name}]  ({args.svg_episodes} episodes × {len(seeds)} seed)\n{'='*50}")
        scores[label] = run_seeded(
            label, out_root, seeds, ["svg_agent.pt"],
            lambda seed, out_dir, curves, weights=weights, label=label: train_svg(
                base_ensemble, dataset, args.svg_episodes, args.svg_horizon,
                args.hidden, weights, out_dir,
                seed=seed, learning_curves=curves, curve_label=label, **common_kwargs,
            ),
            learning_curves,
        )

    print_summary(scores)

    curve_path = save_all_learning_curves_plot(learning_curves, out_root)
    if curve_path is not None:
        print(f"Learning curves saved → {curve_path}")

    summary_path = out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
