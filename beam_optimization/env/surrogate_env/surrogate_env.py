"""
Surrogate ensemble provides SYNTHETIC transitions (cheap, ~1 ms/step, backgroud planning)

Shares its reset/step scaffolding with TraceWinEnv via BaseBeamEnv (env/base_env.py).
The actual ModularMLP forward pass lives in SurrogateBeamSimulator.

State / Observation:
    Beam states flattened into a 1-D vector. The size depends on obs_mode:
        'full'             — all 12 stages (initial + 11 predicted): 12 x 9 = 108 dim
        'final'            — final stage only:                         1 x 9 =   9 dim
        'final_with_beam0' — initial + final stage:                    2 x 9 =  18 dim
    The initial beam state (stage 0) is sampled from the dataset at episode reset and
    stays fixed for the whole episode, giving the agent the physics context.

Action:
    Delta on all 16 parameters: bounded by ±(action_scale x sensitivity).

Reward:
    score(t+1) - score(t) 

Episode design:
    RESET:
        1. Sample beam0 from dataset (or from N(mu, sigma) if beam0_mode='gaussian')
        2. Run surrogate(params) -> beam_states at all 12 stages
        3. obs = flatten(beam_states) -> initial RL state
    STEP:
        params_{t+1} = params_t + action
        surrogate(params_{t+1}) -> obs_{t+1}
        reward = score(t+1) - score(t)

    Truncated after max_steps steps. Never terminated early.
"""
from __future__ import annotations

from typing import List, Optional, Union

from beam_optimization.env.base_beam_env import BaseBeamEnv
from beam_optimization.env.surrogate_env.surrogate_simulator import SurrogateBeamSimulator
from beam_optimization.env.surrogate_env.surrogate.modular_mlp import ModularMLP
from beam_optimization.env.dataset import SurrogateTrainingDataset


class SurrogateEnv(BaseBeamEnv):
    """Fast surrogate-based beam optimization environment.

    Args:
        model:        Trained ModularMLP surrogate (or list for ensemble).
        dataset:      SurrogateTrainingDataset with initial beam states for episode reset.
        action_scale: Multiplier on sensitivity for action bounds.
        max_steps:    Episode length.
        sigma_factor: Gaussian noise scale (x sensitivity) for initial parameters.
        obs_mode:     'full' (108 features = 12 stages x 9 features), 'final' (9 = 1 stage x 9 features), or 'final_with_beam0' (18 = 2 stages x 9 features).
        beam0_mode:   How to sample the initial beam state at each reset:
                        'dataset'  — pick a random row from the dataset (default)
                        'gaussian' — sample from N(μ, sigma) fitted on the dataset
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
        # Store the simulator kwargs for later use in _build_simulator() for the surrogate simulator 
        self._simulator_kwargs = { "model": model, "dataset": dataset, "beam0_mode": beam0_mode, "device": device}
        
        # Call the base class constructor
        super().__init__( action_scale=action_scale, max_steps=max_steps, sigma_factor=sigma_factor, obs_mode=obs_mode)


    def _build_simulator(self) -> SurrogateBeamSimulator:
        return SurrogateBeamSimulator(**self._simulator_kwargs)
