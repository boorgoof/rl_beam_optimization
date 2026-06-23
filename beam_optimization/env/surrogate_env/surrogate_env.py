"""
SurrogateEnv — Gymnasium environment backed by the ModularMLP surrogate.

Used for fast RL training without calling TraceWin. Shares its reset/step
scaffolding with TraceWinEnv via BaseBeamEnv (env/base_env.py); the actual
ModularMLP forward pass lives in SurrogateBeamSimulator.

State / Observation:
    Flattened beam states at all 12 stages (initial + 11 predicted) = 12 × 9 = 108 dim.
    The initial beam state (stage 0) is sampled from the dataset at episode reset and
    stays fixed for the whole episode, giving the agent the physics context.

Action:
    Delta on all 16 parameters: bounded by ±(action_scale × sensitivity).

Reward:
    score(t+1) - score(t)  — purely delta-based, no shaping.

Episode:
    Terminates after max_steps steps (truncated, never terminated).
"""
from __future__ import annotations

from typing import List, Optional, Union

from beam_optimization.env.base_beam_env import BaseBeamEnv
from beam_optimization.env.surrogate_env.simulator import SurrogateBeamSimulator
from beam_optimization.env.surrogate_env.surrogate.modular_mlp import ModularMLP
from beam_optimization.env.surrogate_env.surrogate.dataset import SurrogateTrainingDataset


class SurrogateEnv(BaseBeamEnv):
    """Fast surrogate-based beam optimization environment.

    Args:
        model:        Trained ModularMLP surrogate (or list for ensemble).
        dataset:      SurrogateTrainingDataset with initial beam states for episode reset.
        action_scale: Multiplier on sensitivity for action bounds.
        max_steps:    Episode length.
        sigma_factor: Gaussian noise scale (× sensitivity) for initial parameters.
        obs_mode:     'full' (108), 'final' (9), or 'final_with_beam0' (18).
        beam0_mode:   How to sample the initial beam state at each reset:
                        'dataset'  — pick a random row from the dataset (default)
                        'gaussian' — sample from N(μ, σ) fitted on the dataset
        device:       Torch device for inference.
    """

    def __init__(
        self,
        model: Union[ModularMLP, List[ModularMLP]],
        dataset: SurrogateTrainingDataset,
        action_scale: float = 1.0,
        max_steps: int = 50,
        sigma_factor: float = 0.5,
        obs_mode: str = "full",
        beam0_mode: str = "dataset",
        device: Optional[str] = None,
    ):
        self.simulator = SurrogateBeamSimulator(
            model=model,
            dataset=dataset,
            beam0_mode=beam0_mode,
            device=device,
        )
        # Compatibility for existing exploratory code/tests.
        self._ensemble = self.simulator._ensemble
        self.dataset = self.simulator.dataset
        self.device = self.simulator.device
        self.beam0_mode = self.simulator.beam0_mode

        super().__init__(
            action_scale=action_scale,
            max_steps=max_steps,
            sigma_factor=sigma_factor,
            obs_mode=obs_mode,
        )

    @property
    def model(self):
        return self.simulator.model

    @property
    def _episode_beam0(self):
        return self.simulator._episode_beam0
