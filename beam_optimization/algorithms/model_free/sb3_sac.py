"""Compatibility shim for the former SB3SAC wrapper."""
from __future__ import annotations

from typing import Optional

from beam_optimization.algorithms.model_free.stable_baselines import (
    StableBaselinesAgent,
    _BestScoreCallback,
)

class SB3SAC(StableBaselinesAgent):
    """Deprecated SAC-only adapter retained for old imports."""
    def __init__(
        self,
        env,
        hidden_dims=(256, 256),
        lr: float = 3e-4,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        device: str = "auto",
        seed: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
    ):
        super().__init__(
            "sac",
            env,
            hidden_dims=tuple(hidden_dims),
            device=device,
            seed=seed,
            tensorboard_log=tensorboard_log,
            model_kwargs={
                "learning_rate": lr,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "tau": tau,
                "gamma": gamma,
            },
        )

    @classmethod
    def load(cls, path: str, env) -> "SB3SAC":
        loaded = StableBaselinesAgent.load("sac", path, env)
        instance = cls.__new__(cls)
        instance.__dict__.update(loaded.__dict__)
        return instance
