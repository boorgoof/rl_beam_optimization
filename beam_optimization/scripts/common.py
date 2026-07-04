"""Shared episode/evaluation helpers for the CLI scripts (train/test/benchmark)."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from beam_optimization.config.adige import BEAM_STATE_FEATURES


def select_eval_action(agent, obs):
    """Deterministic action from any agent type (custom, MBPO, SVG, SB3)."""
    try:
        return agent.select_action(obs, training=False)
    except TypeError:
        return agent.select_action(obs, deterministic=True)


def final_features(info: dict) -> dict[str, float]:
    """Extract the final-stage beam features from a step info dict."""
    result = info.get("sim_result")
    if result is None or result.beam_states is None:
        return {}
    final = np.asarray(result.beam_states[-1], dtype=float).reshape(-1)
    return {
        name: float(final[i])
        for i, name in enumerate(BEAM_STATE_FEATURES)
        if i < len(final)
    }


def run_episode(
    env,
    agent,
    *,
    seed: Optional[int] = None,
    reset_options: Optional[dict] = None,
    step_callback: Optional[Callable] = None,
) -> dict:
    """Run one greedy episode and return its summary.

    step_callback(step_idx, env, info, done) is invoked after reset
    (step_idx=0, done=False) and after every env step, so callers can render
    or log without reimplementing the loop.
    """
    obs, info = env.reset(seed=seed, options=reset_options)
    if step_callback is not None:
        step_callback(0, env, info, False)

    total_reward = 0.0
    final_score = float(info.get("score", np.nan))
    features = final_features(info)
    steps: list[dict] = []
    done = False
    step_idx = 0

    while not done:
        action = select_eval_action(agent, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_idx += 1
        total_reward += float(reward)
        final_score = float(info.get("score", final_score))
        features = final_features(info) or features
        steps.append({
            "step": step_idx,
            "reward": float(reward),
            "score": float(info.get("score", np.nan)),
            "action_norm": float(np.linalg.norm(action)),
        })
        if step_callback is not None:
            step_callback(step_idx, env, info, done)

    return {
        "total_reward": float(total_reward),
        "final_score": float(final_score),
        "final_features": features,
        "n_steps": int(step_idx),
        "steps": steps,
    }


def evaluate_policy(agent, make_env: Callable[[], object], n_episodes: int) -> dict[str, float]:
    """Aggregate greedy-episode statistics (used for learning curves and benchmarks)."""
    env = make_env()
    rewards: list[float] = []
    scores: list[float] = []
    try:
        for _ in range(n_episodes):
            result = run_episode(env, agent)
            rewards.append(result["total_reward"])
            scores.append(result["final_score"])
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "best_score": float(np.max(scores)) if scores else 0.0,
        "episodes": int(n_episodes),
    }
