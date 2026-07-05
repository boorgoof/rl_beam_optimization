"""
SB3SAC — wrapper around stable_baselines3.SAC (sanity baseline).

Stable Baselines 3 manages its buffer and optimization internally, so the
API differs from the custom agents (no step-by-step store/optimize).
This wrapper:
  - exposes train(env, n_steps) for the training phase
  - exposes select_action(obs) for deterministic inference
  - exposes save(path) / load(path, env) for checkpoints

Typical use:
    from beam_optimization.algorithms.model_free.sb3_sac import SB3SAC
    env   = SurrogateEnv(model=surrogate, dataset=ds)
    agent = SB3SAC(env, hidden_dims=(256, 256), lr=3e-4)
    best  = agent.train(env, n_steps=200_000)
    agent.save("runs/sb3_sac/model.zip")
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from beam_optimization.algorithms.utils.logger import Logger

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def _check_sb3():
    if not _SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 not found. Install with:\n"
            "  pip install stable-baselines3"
        )


class _BestScoreCallback(BaseCallback):
    """Track the env best score during SB3 training."""

    def __init__(
        self,
        logger: Optional[Logger] = None,
        log_every: int = 10_000,
        agent=None,
        eval_every: int = 1000,
        eval_episodes: int = 5,
        eval_fn: Optional[Callable] = None,
        eval_logger: Optional[Callable] = None,
    ):
        super().__init__()
        self.best_score = -float("inf")
        self.metrics_logger = logger
        self.log_every = max(1, int(log_every))
        self.agent = agent
        self.eval_every = max(1, int(eval_every))
        self.eval_episodes = max(1, int(eval_episodes))
        self.eval_fn = eval_fn
        self.eval_logger = eval_logger
        self._last_eval_step = None
        self._episode_reward = 0.0
        self._episode_count = 0

    def _run_eval(self, step: int) -> None:
        if self.eval_fn is None or self.eval_logger is None or self.agent is None:
            return
        if self._last_eval_step == step:
            return
        metrics = self.eval_fn(self.agent, self.eval_episodes)
        self.eval_logger(step, metrics)
        self._last_eval_step = step

    def _on_training_start(self) -> None:
        self._run_eval(0)

    def _on_step(self) -> bool:
        # SB3 exposes env info through self.locals
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        if len(rewards) > 0:
            self._episode_reward += float(np.asarray(rewards).reshape(-1)[0])
        for info in infos:
            sc = info.get("score")
            if sc is not None and sc > self.best_score:
                self.best_score = sc
            if self.metrics_logger is not None and self.num_timesteps % self.log_every == 0:
                metrics = {
                    "score": float(sc) if sc is not None else 0.0,
                    "best_score": self.best_score,
                    "episode_reward": self._episode_reward,
                    "episode": float(self._episode_count),
                }
                self.metrics_logger.log(metrics, step=self.num_timesteps)
        if len(dones) > 0 and bool(np.asarray(dones).reshape(-1)[0]):
            self._episode_count += 1
            self._episode_reward = 0.0
        if self.num_timesteps % self.eval_every == 0:
            self._run_eval(self.num_timesteps)
        return True


class SB3SAC:
    """stable_baselines3.SAC wrapper for the beam envs.

    Args:
        env:         Gymnasium environment (SurrogateEnv or TraceWinEnv).
        hidden_dims: Hidden layer sizes of the policy/value nets.
        lr:          Adam learning rate.
        buffer_size: SB3 internal replay buffer size.
        batch_size:  Batch size per SAC update.
        tau:         Soft update coefficient (target networks).
        gamma:       Discount factor.
        device:      'cpu', 'cuda', or 'auto'.
    """

    def __init__(
        self,
        env,
        hidden_dims: Tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        device: str = "auto",
        seed: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
    ):
        _check_sb3()
        policy_kwargs = {"net_arch": list(hidden_dims)}
        self._model = SAC(
            "MlpPolicy",
            env,
            learning_rate=lr,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
            tensorboard_log=tensorboard_log,
            verbose=0,
        )
        self._env = env

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, env=None, n_steps: int = 200_000,
              log_every: int = 10_000,
              logger: Optional[Logger] = None,
              eval_every: int = 1000,
              eval_episodes: int = 5,
              eval_fn: Optional[Callable] = None,
              eval_logger: Optional[Callable] = None) -> float:
        """Run SB3 training for n_steps real env steps.

        Args:
            env:       If given, replaces the env from the constructor.
            n_steps:   Total number of env interaction steps.
            log_every: Print a summary every N steps.

        Returns:
            Best score reached during training.
        """
        _check_sb3()
        if env is not None and env is not self._env:
            self._model.set_env(env)

        cb = _BestScoreCallback(
            logger=logger,
            log_every=log_every,
            agent=self,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_fn=eval_fn,
            eval_logger=eval_logger,
        )
        self._model.learn(
            total_timesteps=n_steps,
            callback=cb,
            log_interval=max(1, log_every // 1000),
            reset_num_timesteps=True,
        )
        return cb.best_score

    # ── Inference ─────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray,
                      deterministic: bool = True) -> np.ndarray:
        """Select an action (deterministic by default)."""
        _check_sb3()
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return action

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save the SB3 model. SB3 automatically appends the .zip extension."""
        _check_sb3()
        self._model.save(path)

    @classmethod
    def load(cls, path: str, env) -> "SB3SAC":
        """Load a saved SB3 model."""
        _check_sb3()
        instance = cls.__new__(cls)
        instance._model = SAC.load(path, env=env)
        instance._env   = env
        return instance
