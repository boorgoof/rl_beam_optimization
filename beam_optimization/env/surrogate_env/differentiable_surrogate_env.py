"""Differentiable surrogate rollout API for SVG.

This module keeps the Gym contract inherited from SurrogateEnv untouched and
adds a torch-only API for algorithms that need autograd through the surrogate.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator
from typing import List, Optional, Union

import numpy as np
import torch

from beam_optimization.config.adige import (
    ERROR_SCORE,
    LOW_TRANSMISSION_REWARD,
    MAX_STEPS,
    N_OUTPUT_STAGES,
    N_PARAMS,
    REWARD_SCORE_SCALE,
    STAGE_PARAM_SIZES,
    TEST_RESET_SCALE,
    TRAIN_RESET_SCALE,
    action_step_vec,
    clip_param_tensor_to_hw,
    default_params,
    params_to_vec,
    reset_std_vec,
    score_tensor,
    select_observation_stages_tensor,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.surrogate_env.surrogate_env import SurrogateEnv


@dataclass
class DifferentiableBeamState:
    """Torch rollout state carried explicitly by SVG."""

    beam0: torch.Tensor
    params: torch.Tensor
    obs: torch.Tensor
    score: torch.Tensor
    beam_states: List[torch.Tensor]
    step_count: int
    model_index: int

    def detach_for_next_step(self) -> "DifferentiableBeamState":
        """copy the state and detach all tensors from the current autograd graph"""
        return DifferentiableBeamState(
            beam0=self.beam0.detach(),
            params=self.params.detach(),
            obs=self.obs.detach(),
            score=self.score.detach(),
            beam_states=[stage.detach() for stage in self.beam_states],
            step_count=self.step_count,
            model_index=self.model_index,
        )


class DifferentiableSurrogateEnv(SurrogateEnv):
    """SurrogateEnv plus a torch/autograd rollout API for SVG.

    reset()/step() remain the inherited Gym/numpy API. reset_torch()/step_torch()
    use an explicit DifferentiableBeamState and do not mutate the Gym episode
    fields such as _current_params, _current_obs, or _current_score.
    """

    def __init__(
        self,
        model: Union[ModularMLP, List[ModularMLP]],
        dataset: BeamDataset,
        max_steps: int = MAX_STEPS,
        device: Optional[str] = None,
        stage_weights: Optional[List[float]] = None,
        reset_scale: float = TRAIN_RESET_SCALE,
        recovery_reset_probability: float = 0.0,
        recovery_reset_scale: float = TEST_RESET_SCALE,
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            max_steps=max_steps,
            device=device,
            reset_scale=reset_scale,
            recovery_reset_probability=recovery_reset_probability,
            recovery_reset_scale=recovery_reset_scale,
        )
        self.device = self.simulator.device
        self._reset_std_t = torch.tensor(
            reset_std_vec(reset_scale), dtype=torch.float32, device=self.device
        )
        self._recovery_reset_std_t = torch.tensor(
            reset_std_vec(recovery_reset_scale),
            dtype=torch.float32,
            device=self.device,
        )
        self._action_step_t = torch.tensor(
            action_step_vec(), dtype=torch.float32, device=self.device
        )
        self._defaults_t = torch.tensor(
            params_to_vec(default_params()), dtype=torch.float32, device=self.device
        )
        self._stage_weights_t = self._build_stage_weights(stage_weights)

    @contextmanager
    def frozen_surrogate_weights(self) -> Iterator[None]:
        """Temporarily freeze active surrogate weights while preserving input gradients.

        SVG needs gradients through the surrogate forward pass back to the
        action/policy, but it must not accumulate gradients on the surrogate
        weights themselves. This context manager changes only parameter
        requires_grad flags; it does not use torch.no_grad().
        """
        params = list(self.simulator.model.parameters())
        previous_flags = [param.requires_grad for param in params]
        try:
            for param in params:
                param.requires_grad_(False)
            yield
        finally:
            for param, requires_grad in zip(params, previous_flags):
                param.requires_grad_(requires_grad)

    def reset_torch(
        self,
        beam0: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> DifferentiableBeamState:
        """Start a differentiable surrogate episode without touching Gym state."""
        model_index = self.simulator.sample_model_index()
        self.simulator.set_active_model(model_index)

        beam0_t = self._prepare_beam0(beam0)
        use_recovery_reset = (
            self.recovery_reset_probability > 0.0
            and float(torch.rand((), device=self.device))
            < self.recovery_reset_probability
        )
        reset_std = (
            self._recovery_reset_std_t if use_recovery_reset else self._reset_std_t
        )
        params = clip_param_tensor_to_hw(
            self._defaults_t + torch.randn(N_PARAMS, device=self.device) * reset_std
        ).detach()
        beam_states = self._forward(params, beam0_t)
        score = self._score_beam_states(beam_states).detach()
        obs = self._build_obs(beam0_t, beam_states).detach()

        return DifferentiableBeamState(
            beam0=beam0_t.detach(),
            params=params,
            obs=obs,
            score=score,
            beam_states=[stage.detach() for stage in beam_states],
            step_count=0,
            model_index=model_index,
        )

    def step_torch(
        self,
        state: DifferentiableBeamState,
        action: torch.Tensor,
    ) -> tuple[DifferentiableBeamState, torch.Tensor]:
        """Apply one differentiable action and return (next_state, reward)."""
        self.simulator.set_active_model(state.model_index)

        action = action.to(device=self.device, dtype=torch.float32)
        if action.dim() == 2:
            if action.shape[0] != 1:
                raise ValueError(f"SVG torch actions must use batch size 1, got {tuple(action.shape)}")
            action = action.squeeze(0)
        if action.shape != (N_PARAMS,):
            raise ValueError(f"SVG torch action must have shape ({N_PARAMS},), got {tuple(action.shape)}")

        # Same action-box clip as BaseBeamEnv.step (differentiable clamp; the
        # tanh policy already respects the bounds, so this rarely binds).
        action = torch.clamp(action, -self._action_step_t, self._action_step_t)
        params_next = clip_param_tensor_to_hw(state.params + action)
        beam_states = self._forward(params_next, state.beam0)
        score_next = self._score_beam_states(beam_states)
        reward = torch.where(
            score_next == ERROR_SCORE,
            score_next.new_full((), LOW_TRANSMISSION_REWARD),
            score_next / REWARD_SCORE_SCALE,
        )
        obs_next = self._build_obs(state.beam0, beam_states)

        next_state = DifferentiableBeamState(
            beam0=state.beam0,
            params=params_next,
            obs=obs_next,
            score=score_next,
            beam_states=beam_states,
            step_count=state.step_count + 1,
            model_index=state.model_index,
        )
        return next_state, reward

    def _prepare_beam0(
        self,
        beam0: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> torch.Tensor:
        if beam0 is None:
            beam0 = self.simulator.sample_beam0()
        if isinstance(beam0, torch.Tensor):
            beam0_t = beam0.to(device=self.device, dtype=torch.float32)
        else:
            beam0_t = torch.tensor(beam0, dtype=torch.float32, device=self.device)
        if beam0_t.dim() == 1:
            beam0_t = beam0_t.unsqueeze(0)
        if beam0_t.dim() != 2 or beam0_t.shape[0] != 1:
            raise ValueError(f"beam0 must have shape (9,) or (1, 9), got {tuple(beam0_t.shape)}")
        return beam0_t

    def _split_params_grad(self, params: torch.Tensor) -> List[torch.Tensor]:
        tensors = []
        offset = 0
        for size in STAGE_PARAM_SIZES:
            tensors.append(params[offset:offset + size].unsqueeze(0))
            offset += size
        return tensors

    def _forward(self, params: torch.Tensor, beam0: torch.Tensor) -> List[torch.Tensor]:
        return self.simulator.forward_differentiable(
            self.simulator.model,
            beam0,
            self._split_params_grad(params),
        )

    def _build_obs(
        self,
        beam0: torch.Tensor,
        outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        return select_observation_stages_tensor([beam0] + outputs)

    def _score_beam_states(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        if self._stage_weights_t is None:
            return score_tensor(outputs[-1])
        scores = torch.stack([score_tensor(stage) for stage in outputs], dim=0)
        return (scores * self._stage_weights_t.view(-1, 1)).sum(dim=0)

    def _build_stage_weights(self, stage_weights: Optional[List[float]]) -> Optional[torch.Tensor]:
        if stage_weights is None:
            return None
        if len(stage_weights) != N_OUTPUT_STAGES:
            raise ValueError(
                f"stage_weights must have length {N_OUTPUT_STAGES}, got {len(stage_weights)}"
            )
        weights = torch.tensor(stage_weights, dtype=torch.float32, device=self.device)
        weight_sum = weights.sum()
        if float(weight_sum) == 0.0:
            raise ValueError("stage_weights must not sum to zero")
        return weights / weight_sum
