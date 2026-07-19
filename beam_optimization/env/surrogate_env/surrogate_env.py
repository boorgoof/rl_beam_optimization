"""
Surrogate ensemble provides SYNTHETIC transitions (cheap, ~1 ms/step, background planning)

Shares its reset/step scaffolding with TraceWinEnv via BaseBeamEnv (env/base_beam_env.py).
The actual ModularMLP forward pass lives in SurrogateBeamSimulator.

State / Observation:
    Beam states selected by OBSERVATION_STAGE_MASK in adige.py and flattened
    into a 1-D vector.
    The initial beam state (stage 0) is sampled from the dataset at episode reset and
    stays fixed for the whole episode, giving the agent the physics context.

Action:
    Delta on all configured parameters, bounded by per-parameter action_step_vec().

Reward:
    score(t+1) - score(t) 

Episode design:
    RESET:
        1. Sample beam0 from the dataset
        2. Run surrogate(params) -> beam_states at all 12 stages
        3. obs = selected/flattened beam_states -> initial RL state
    STEP:
        params_{t+1} = params_t + action
        surrogate(params_{t+1}) -> obs_{t+1}
        reward = score(t+1) - score(t)

    Truncated after max_steps steps. Terminated early only if the simulator
    reports a failure (bounded FAILURE_PENALTY reward, see BaseBeamEnv.step).
"""
from __future__ import annotations

from typing import List, Optional, Union

from beam_optimization.config.adige import MAX_STEPS
from beam_optimization.env.base_beam_env import BaseBeamEnv
from beam_optimization.env.surrogate_env.surrogate.surrogate_simulator import (
    SurrogateBeamSimulator,
)
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.dataset import BeamDataset


class SurrogateEnv(BaseBeamEnv):
    """Fast surrogate-based beam optimization environment.

    Args:
        model:        Trained ModularMLP surrogate (or list for ensemble).
        dataset:      BeamDataset with initial beam states for episode reset.
        max_steps:    Episode length.
        observation:   Selected by OBSERVATION_STAGE_MASK in adige.py.
        device:       Torch device for inference.
    """

    def __init__(
        self,
        model: Union[ModularMLP, List[ModularMLP]],
        dataset: BeamDataset,
        max_steps: int = MAX_STEPS,
        device: Optional[str] = None,
        simulator_seed: Optional[int] = None,
    ):
        # Store the simulator kwargs for later use in _build_simulator() for the surrogate simulator
        self._simulator_kwargs = {
            "model": model,
            "dataset": dataset,
            "device": device,
            "seed": simulator_seed,
        }
        
        # Call the base class constructor
        super().__init__(max_steps=max_steps)


    def _build_simulator(self) -> SurrogateBeamSimulator:
        return SurrogateBeamSimulator(**self._simulator_kwargs)

    def sample_active_model(self) -> int:
        """Sample and activate one surrogate ensemble member."""
        if hasattr(self.simulator, "set_active_model"):
            index = self.simulator.sample_model_index(self.np_random)
            self.simulator.set_active_model(index)
            return index
        return 0
