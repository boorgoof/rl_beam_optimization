"""Unified wrappers for continuous-action Stable Baselines3 algorithms."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from beam_optimization.algorithms import (
    STABLE_BASELINES_ALGORITHMS,
    canonical_algorithm_name,
)
from beam_optimization.algorithms.utils.atomic_save import atomic_save
from beam_optimization.algorithms.utils.logger import Logger

try:
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import FloatSchedule

    _SB3_CLASSES = {
        "sac": SAC,
        "ppo": PPO,
        "td3": TD3,
        "ddpg": DDPG,
        "a2c": A2C,
    }
    _SB3_AVAILABLE = True
except ImportError:
    BaseCallback = object
    _SB3_CLASSES = {}
    _SB3_AVAILABLE = False


def _check_sb3() -> None:
    if not _SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 not found. Install with:\n"
            "  pip install stable-baselines3"
        )


class _BestScoreCallback(BaseCallback):
    """Track physical score, training rewards, and periodic evaluations."""

    def __init__(
        self,
        logger: Optional[Logger] = None,
        log_every: int = 10_000,
        agent=None,
        eval_every: int = 1000,
        eval_episodes: int = 5,
        eval_fn: Optional[Callable] = None,
        eval_logger: Optional[Callable] = None,
        step_hook: Optional[Callable[[int, int, list[dict]], None]] = None,
        initial_eval_step: int = 0,
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
        self.step_hook = step_hook
        self.initial_eval_step = int(initial_eval_step)
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
        self._run_eval(self.initial_eval_step)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        if len(rewards) > 0:
            self._episode_reward += float(np.asarray(rewards).reshape(-1)[0])
        for info in infos:
            score = info.get("score")
            if score is not None and score > self.best_score:
                self.best_score = float(score)
            if self.metrics_logger is not None and self.num_timesteps % self.log_every == 0:
                self.metrics_logger.log(
                    {
                        "score": float(score) if score is not None else 0.0,
                        "best_score": self.best_score,
                        "episode_reward": self._episode_reward,
                        "episode": float(self._episode_count),
                        "distance_penalty": float(info.get("distance_penalty", 0.0)),
                        "action_penalty": float(info.get("action_penalty", 0.0)),
                        "action_smoothness_penalty": float(
                            info.get("action_smoothness_penalty", 0.0)
                        ),
                        "score_regression_penalty": float(
                            info.get("score_regression_penalty", 0.0)
                        ),
                    },
                    step=self.num_timesteps,
                )
        if len(dones) > 0 and bool(np.asarray(dones).reshape(-1)[0]):
            self._episode_count += 1
            self._episode_reward = 0.0
        if self.num_timesteps % self.eval_every == 0:
            self._run_eval(self.num_timesteps)
        if self.step_hook is not None:
            self.step_hook(self.n_calls, self.num_timesteps, list(infos))
        return True


class StableBaselinesAgent:
    """Common project adapter for SAC, PPO, TD3, DDPG, and A2C from SB3."""

    def __init__(
        self,
        algorithm: str,
        env,
        hidden_dims: Tuple[int, ...] = (256, 256),
        device: str = "auto",
        seed: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
        model_kwargs: Optional[Mapping] = None,
    ):
        _check_sb3()
        algorithm = canonical_algorithm_name(algorithm)
        if algorithm not in STABLE_BASELINES_ALGORITHMS:
            raise ValueError(
                f"Unknown Stable Baselines3 algorithm '{algorithm}'. Available: "
                f"{', '.join(STABLE_BASELINES_ALGORITHMS)}"
            )
        hidden = list(hidden_dims)
        if algorithm in {"ppo", "a2c"}:
            net_arch = {"pi": hidden, "vf": hidden}
        else:
            net_arch = hidden
        kwargs = {
            "policy_kwargs": {"net_arch": net_arch},
            "device": device,
            "seed": seed,
            "tensorboard_log": tensorboard_log,
            "verbose": 0,
        }
        if model_kwargs:
            kwargs.update(dict(model_kwargs))

        self.algorithm = algorithm
        self._model = _SB3_CLASSES[algorithm]("MlpPolicy", env, **kwargs)
        self._env = env

    def train(
        self,
        env=None,
        n_steps: int = 200_000,
        log_every: int = 10_000,
        logger: Optional[Logger] = None,
        eval_every: int = 1000,
        eval_episodes: int = 5,
        eval_fn: Optional[Callable] = None,
        eval_logger: Optional[Callable] = None,
        reset_num_timesteps: bool = True,
        step_hook: Optional[Callable[[int, int, list[dict]], None]] = None,
    ) -> float:
        """Train with SB3's native optimization loop."""
        _check_sb3()
        if env is not None and env is not self._env:
            self._model.set_env(env)
            self._env = env
        callback = _BestScoreCallback(
            logger=logger,
            log_every=log_every,
            agent=self,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_fn=eval_fn,
            eval_logger=eval_logger,
            step_hook=step_hook,
            initial_eval_step=0 if reset_num_timesteps else self.num_timesteps,
        )
        self._model.learn(
            total_timesteps=int(n_steps),
            callback=callback,
            log_interval=max(1, int(log_every) // 1000),
            reset_num_timesteps=bool(reset_num_timesteps),
        )
        return callback.best_score

    @property
    def num_timesteps(self) -> int:
        return int(self._model.num_timesteps)

    def set_env(self, env) -> None:
        """Switch Gym environments without recreating SAC or its optimizers."""
        self._model.set_env(env)
        self._env = env

    def reset_replay_buffer(self) -> None:
        """Discard replay transitions while preserving all learned weights."""
        replay_buffer = getattr(self._model, "replay_buffer", None)
        if replay_buffer is None:
            raise RuntimeError("Stable Baselines3 replay buffer is not initialized")
        replay_buffer.reset()

    def delay_learning(self, additional_steps: int) -> int:
        """Delay off-policy gradient updates by N additional environment steps."""
        additional_steps = int(additional_steps)
        if additional_steps < 0:
            raise ValueError("additional_steps must be non-negative")
        target = self.num_timesteps + additional_steps
        self._model.learning_starts = target
        return target

    def set_learning_starts_timestep(self, timestep: int) -> None:
        """Restore an absolute learning-start threshold from workflow state."""
        timestep = int(timestep)
        if timestep < 0:
            raise ValueError("timestep must be non-negative")
        self._model.learning_starts = timestep

    def configure_off_policy_updates(
        self,
        *,
        learning_rate: float,
        gradient_steps: int = 1,
    ) -> None:
        """Change off-policy optimizer settings without recreating the model.

        Iterative Sim-to-Real uses the normal SAC rate on the surrogate and a
        much smaller rate on TraceWin.  Updating both ``learning_rate`` and
        ``lr_schedule`` is required because SB3 reapplies the schedule to the
        actor, critic, and entropy optimizers before every gradient block.
        ``gradient_steps=0`` is intentionally supported so a callback can
        collect an exact number of environment steps while enabling one
        update only at a configured real-step interval.
        """
        if self.algorithm not in {"sac", "td3", "ddpg"}:
            raise TypeError(
                "configure_off_policy_updates is available only for "
                "off-policy algorithms"
            )
        learning_rate = float(learning_rate)
        gradient_steps = int(gradient_steps)
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if gradient_steps < 0:
            raise ValueError("gradient_steps must be non-negative")
        self._model.learning_rate = learning_rate
        self._model.lr_schedule = FloatSchedule(learning_rate)
        self._model.gradient_steps = gradient_steps

    def set_off_policy_gradient_steps(self, gradient_steps: int) -> None:
        """Set updates after the next rollout while preserving exact step budgets."""
        if self.algorithm not in {"sac", "td3", "ddpg"}:
            raise TypeError(
                "set_off_policy_gradient_steps is available only for "
                "off-policy algorithms"
            )
        gradient_steps = int(gradient_steps)
        if gradient_steps < 0:
            raise ValueError("gradient_steps must be non-negative")
        self._model.gradient_steps = gradient_steps

    def save_replay_buffer(self, path: str | Path) -> None:
        """Persist replay separately from the SB3 policy archive."""
        atomic_save(path, self._model.save_replay_buffer)

    def load_replay_buffer(self, path: str | Path) -> None:
        self._model.load_replay_buffer(str(path))

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        # SB3's own .save() writes the zip archive straight to `path`; a
        # concurrent reader (e.g. this same checkpoint reloaded moments
        # later in a notebook) can catch it mid-write and see a truncated
        # zip. Save to a temp file next to `path` first, then atomically
        # replace -- readers only ever see the complete previous or new file.
        #
        # SB3 appends ".zip" itself when given a suffix-less path (callers
        # rely on this, e.g. save("agent") -> reads back "agent.zip"); our
        # temp file always has a suffix (so SB3 never touches it), so that
        # inference has to happen here instead, before the atomic replace
        # target is decided.
        target = Path(path)
        if target.suffix == "":
            target = target.with_suffix(".zip")
        atomic_save(target, self._model.save)

    @classmethod
    def load(cls, algorithm: str, path: str, env) -> "StableBaselinesAgent":
        _check_sb3()
        algorithm = canonical_algorithm_name(algorithm)
        if algorithm not in STABLE_BASELINES_ALGORITHMS:
            raise ValueError(f"Unknown Stable Baselines3 algorithm '{algorithm}'")
        instance = cls.__new__(cls)
        instance.algorithm = algorithm
        instance._model = _SB3_CLASSES[algorithm].load(path, env=env)
        instance._env = env
        return instance
