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
    BEAM_STATE_FEATURES,
    ERROR_SCORE,
    MAX_TERMINAL_RESET_ATTEMPTS,
    MAX_STEPS,
    RL_MIN_NPART_RATIO,
    N_OUTPUT_STAGES,
    N_PARAMS,
    REWARD_SCORE_SCALE,
    STAGE_PARAM_SIZES,
    TERMINAL_FAILURE_REWARD,
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
        distance_penalty_weight: float = 0.0,
        action_penalty_weight: float = 0.0,
        score_regression_penalty_weight: float = 0.0,
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            max_steps=max_steps,
            device=device,
            reset_scale=reset_scale,
            distance_penalty_weight=distance_penalty_weight,
            action_penalty_weight=action_penalty_weight,
            score_regression_penalty_weight=score_regression_penalty_weight,
        )
        self.device = self.simulator.device
        self._reset_std_t = torch.tensor(
            reset_std_vec(reset_scale), dtype=torch.float32, device=self.device
        )
        self._action_step_t = torch.tensor(
            action_step_vec(), dtype=torch.float32, device=self.device
        )
        self._defaults_t = torch.tensor(
            params_to_vec(default_params()), dtype=torch.float32, device=self.device
        )
        self._stage_weights_t = self._build_stage_weights(stage_weights)
        self._knn_reference_t = None
        self._knn_std_t = None
        if self.distance_penalty_weight > 0.0:
            self._knn_reference_t = dataset.get_param_vecs().to(
                device=self.device, dtype=torch.float32
            )
            self._knn_std_t = torch.tensor(
                dataset.param_knn_std(),
                dtype=torch.float32,
                device=self.device,
            )

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
        for _ in range(MAX_TERMINAL_RESET_ATTEMPTS):
            params = clip_param_tensor_to_hw(
                self._defaults_t
                + torch.randn(N_PARAMS, device=self.device) * self._reset_std_t
            ).detach()
            beam_states = self._forward(params, beam0_t)
            if not bool(self._terminal_failure_mask(beam_states).item()):
                break
        else:
            raise RuntimeError(
                "Could not sample a non-terminal differentiable initial state "
                f"after {MAX_TERMINAL_RESET_ATTEMPTS} attempts."
            )
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
    ) -> tuple[DifferentiableBeamState, torch.Tensor, bool]:
        """Apply one action and return (next_state, reward, terminated)."""
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
        terminal_mask = self._terminal_failure_mask(beam_states)
        regular_score = self._score_beam_states(beam_states)
        score_next = torch.where(
            terminal_mask,
            regular_score.new_full((), ERROR_SCORE),
            regular_score,
        )
        terminated = bool(terminal_mask.detach().item())
        if terminated:
            zero = score_next.new_zeros(())
            distance_penalty, action_penalty, regression_penalty = zero, zero, zero
        else:
            distance_penalty, action_penalty, regression_penalty = self._differentiable_penalties(
                state=state,
                action=action,
                params_next=params_next,
                score_next=score_next,
            )
        regular_reward = (
            score_next / REWARD_SCORE_SCALE
            - distance_penalty
            - action_penalty
            - regression_penalty
        )
        reward = torch.where(
            terminal_mask,
            score_next.new_full((), TERMINAL_FAILURE_REWARD),
            regular_reward,
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
        return next_state, reward, terminated

    def _differentiable_penalties(
        self,
        state: DifferentiableBeamState,
        action: torch.Tensor,
        params_next: torch.Tensor,
        score_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Torch equivalents of BaseBeamEnv's three reward regularizers.

        The k-d tree selects neighbors from a detached query. Their distances
        are then recomputed in torch, so the value matches the normal Gym
        environment while gradients flow locally through ``params_next``.
        """
        zero = score_next.new_zeros(())

        distance_penalty = zero
        if self.distance_penalty_weight > 0.0:
            _, indices = self.simulator.dataset.query_param_neighbors(
                params_next.detach().cpu().numpy()[None, :],
                k=5,
            )
            neighbor_indices = torch.as_tensor(
                indices[0], dtype=torch.long, device=self.device
            )
            neighbors = self._knn_reference_t.index_select(0, neighbor_indices)
            standardized_delta = (params_next.unsqueeze(0) - neighbors) / self._knn_std_t
            distance = torch.linalg.vector_norm(standardized_delta, dim=1).mean()
            distance_penalty = self.distance_penalty_weight * distance

        action_penalty = zero
        if self.action_penalty_weight > 0.0:
            normalized_action = torch.where(
                self._action_step_t > 0.0,
                action / self._action_step_t,
                torch.zeros_like(action),
            )
            action_penalty = self.action_penalty_weight * normalized_action.square().mean()

        regression_penalty = zero
        if self.score_regression_penalty_weight > 0.0:
            normalized_drop = torch.relu(state.score - score_next) / REWARD_SCORE_SCALE
            regression_penalty = self.score_regression_penalty_weight * normalized_drop

        return distance_penalty, action_penalty, regression_penalty

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

    @staticmethod
    def _terminal_failure_mask(outputs: List[torch.Tensor]) -> torch.Tensor:
        npart_index = BEAM_STATE_FEATURES.index("npart_ratio")
        return outputs[-1][:, npart_index] < RL_MIN_NPART_RATIO

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
