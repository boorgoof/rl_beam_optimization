"""Evaluate one trained policy on SurrogateEnv or TraceWinEnv.

This command is intentionally separate from benchmark:
benchmark compares many methods numerically on the surrogate, while evaluate
inspects one trained policy step by step and can save render images.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from beam_optimization.config.adige import N_PARAMS, action_bounds
from beam_optimization.config.paths import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SURROGATE_DIR,
    DEFAULT_TRACEWIN_INI,
    default_eval_calc_dir,
)
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP


ACT_DIM = N_PARAMS


def load_surrogate(path: str | Path):
    """Load one ModularMLP or an ensemble from a directory."""
    path = Path(path)
    if path.is_dir():
        model_files = sorted(path.glob("surrogate_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No surrogate_*.pt files found in {path}")
        models = [ModularMLP.load(str(p)) for p in model_files]
        for model in models:
            model.eval()
        return models

    model = ModularMLP.load(str(path))
    model.eval()
    return model


def make_env(args):
    """Create the selected evaluation environment."""
    if args.env == "surrogate":
        surrogate = load_surrogate(args.surrogate)
        dataset = BeamDataset.load(args.dataset)
        return SurrogateEnv(
            model=surrogate,
            dataset=dataset,
            max_steps=args.max_ep_steps,
        )

    from beam_optimization.env.tracewin_env import TraceWinEnv

    project_file = Path(args.tracewin_project)
    calc_dir = Path(args.calc_dir) if args.calc_dir else default_eval_calc_dir(project_file)
    return TraceWinEnv(
        project_file=str(project_file),
        calc_dir=str(calc_dir),
        max_steps=args.max_ep_steps,
        timeout=args.tracewin_timeout,
    )


def make_agent(algo: str, policy_path: str, obs_dim: int, hidden: list[int], env=None):
    """Instantiate and load a trained policy."""
    bounds = action_bounds()
    action_bounds_tuple = (bounds[0].tolist(), bounds[1].tolist())

    if algo == "sac":
        from beam_optimization.algorithms.model_free.sac import SAC
        agent = SAC(obs_dim, ACT_DIM, action_bounds_tuple, hidden_dims=tuple(hidden))
        agent.load(policy_path)
        return agent
    if algo == "td3":
        from beam_optimization.algorithms.model_free.td3 import TD3
        agent = TD3(obs_dim, ACT_DIM, action_bounds_tuple, hidden_dims=tuple(hidden))
        agent.load(policy_path)
        return agent
    if algo == "ppo":
        from beam_optimization.algorithms.model_free.ppo import PPO
        agent = PPO(obs_dim, ACT_DIM, action_bounds_tuple, hidden_dims=tuple(hidden))
        agent.load(policy_path)
        return agent
    if algo == "ddpg":
        from beam_optimization.algorithms.model_free.ddpg import DDPG
        agent = DDPG(obs_dim, ACT_DIM, action_bounds_tuple, hidden_dims=tuple(hidden))
        agent.load(policy_path)
        return agent
    if algo == "a2c":
        from beam_optimization.algorithms.model_free.a2c import A2C
        agent = A2C(obs_dim, ACT_DIM, action_bounds_tuple, hidden_dims=tuple(hidden))
        agent.load(policy_path)
        return agent
    if algo == "reinforce":
        from beam_optimization.algorithms.model_free.reinforce import REINFORCE
        agent = REINFORCE(obs_dim, ACT_DIM, action_bounds_tuple, hidden_dims=tuple(hidden))
        agent.load(policy_path)
        return agent
    if algo == "trpo":
        from beam_optimization.algorithms.model_free.trpo import TRPO
        agent = TRPO(obs_dim, ACT_DIM, action_bounds_tuple, hidden_dims=tuple(hidden))
        agent.load(policy_path)
        return agent
    if algo == "sb3_sac":
        from beam_optimization.algorithms.model_free.sb3_sac import SB3SAC
        if env is None:
            raise ValueError("SB3 SAC loading requires the evaluation env.")
        return SB3SAC.load(policy_path, env=env)

    raise ValueError(f"Unsupported algo: {algo}")


def select_eval_action(agent, algo: str, obs: np.ndarray) -> np.ndarray:
    if algo == "sb3_sac":
        return agent.select_action(obs, deterministic=True)
    return agent.select_action(obs, training=False)


def final_features(info: dict[str, Any]) -> dict[str, float]:
    result = info.get("sim_result")
    if result is None or result.beam_states is None:
        return {}

    from beam_optimization.config.adige import BEAM_STATE_FEATURES

    final = np.asarray(result.beam_states[-1], dtype=float)
    return {name: float(final[i]) for i, name in enumerate(BEAM_STATE_FEATURES)}


def save_render(env, args, episode_idx: int, step_idx: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    render_dir = Path(args.render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    prefix = render_dir / f"{args.env}_{args.algo}_ep{episode_idx:03d}_step{step_idx:03d}"

    if args.env == "tracewin":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
            fig = env.render(render_beam_distribution=False)
        fig.savefig(prefix.with_name(prefix.name + "_features.png"), dpi=args.dpi)
        plt.close(fig)

        if args.tracewin_phase_space:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
                phase_fig = env.render_final_beam_distribution(
                    max_particles=args.max_particles,
                    bins=args.bins,
                )
            if phase_fig is not None:
                phase_fig.savefig(prefix.with_name(prefix.name + "_phase_space.png"), dpi=args.dpi)
                plt.close(phase_fig)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
            fig = env.render()
        fig.savefig(prefix.with_suffix(".png"), dpi=args.dpi)
        plt.close(fig)


def run_episode(env, agent, args, episode_idx: int) -> dict[str, Any]:
    obs, info = env.reset(seed=args.seed + episode_idx)
    if args.render:
        save_render(env, args, episode_idx, 0)

    episode_reward = 0.0
    steps = []
    done = False
    step_idx = 0

    while not done:
        action = select_eval_action(agent, args.algo, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_idx += 1
        episode_reward += float(reward)

        row = {
            "step": step_idx,
            "reward": float(reward),
            "score": float(info.get("score", np.nan)),
            "prev_score": float(info.get("prev_score", np.nan)),
            "action_norm": float(np.linalg.norm(action)),
            "features": final_features(info),
        }
        steps.append(row)

        print(
            f"  ep={episode_idx} step={step_idx:02d} "
            f"reward={row['reward']:.4g} score={row['score']:.4g} "
            f"|action|={row['action_norm']:.4g}"
        )

        if args.render and (step_idx % args.render_every == 0 or done):
            save_render(env, args, episode_idx, step_idx)

    return {
        "episode": episode_idx,
        "total_reward": float(episode_reward),
        "final_score": float(info.get("score", np.nan)),
        "n_steps": step_idx,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate one trained policy and optionally render episodes.")
    parser.add_argument("--algo", required=True,
                        choices=["sac", "td3", "ppo", "ddpg", "a2c", "reinforce", "trpo", "sb3_sac"])
    parser.add_argument("--policy", required=True, help="Path to the trained policy checkpoint.")
    parser.add_argument("--env", default="surrogate", choices=["surrogate", "tracewin"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-ep-steps", type=int, default=20)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--surrogate", default=str(DEFAULT_SURROGATE_DIR))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--tracewin-project", default=str(DEFAULT_TRACEWIN_INI))
    parser.add_argument("--calc-dir", default=None)
    parser.add_argument("--tracewin-timeout", type=float, default=120.0)

    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "evaluation.json"))
    parser.add_argument("--render", action="store_true", help="Save render PNG files during evaluation.")
    parser.add_argument("--render-dir", default=str(DEFAULT_OUTPUT_DIR / "renders"))
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--tracewin-phase-space", action=argparse.BooleanOptionalAction, default=True,
                        help="For TraceWin render, also save true final phase-space images from .dst files.")
    parser.add_argument("--max-particles", type=int, default=40000)
    parser.add_argument("--bins", type=int, default=150)
    args = parser.parse_args()

    if not Path(args.policy).exists():
        raise FileNotFoundError(args.policy)

    env = make_env(args)
    obs_dim = env.observation_space.shape[0]
    agent = make_agent(args.algo, args.policy, obs_dim, args.hidden, env=env)

    print(f"Environment: {args.env}")
    print(f"Policy:      {args.policy}")
    print(f"Observation: configured mask ({obs_dim} dims)")
    if args.render:
        print(f"Render dir:  {args.render_dir}")

    results = []
    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}")
        results.append(run_episode(env, agent, args, ep))

    summary = {
        "algo": args.algo,
        "policy": args.policy,
        "env": args.env,
        "observation_dim": obs_dim,
        "episodes": results,
        "mean_final_score": float(np.mean([r["final_score"] for r in results])),
        "best_final_score": float(np.max([r["final_score"] for r in results])),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEVALUATION SUMMARY")
    print(f"  mean_final_score: {summary['mean_final_score']:.6g}")
    print(f"  best_final_score: {summary['best_final_score']:.6g}")
    print(f"  saved: {output}")


if __name__ == "__main__":
    main()
