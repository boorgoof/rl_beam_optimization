"""Differentiable surrogate rollout API for SVG.

This module keeps the Gym contract inherited from SurrogateEnv untouched and
adds a torch-only API for algorithms that need autograd through the surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass
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
from beam_optimization.env.simulation import DifferentiableBeamSimulationResult
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.surrogate_env.surrogate_env import SurrogateEnv


@dataclass
class DifferentiableEpisodeState:
    """Torch rollout state carried explicitly by SVG."""

    simulation: DifferentiableBeamSimulationResult
    obs: torch.Tensor
    score: torch.Tensor
    step_count: int
    previous_action: Optional[torch.Tensor] = None

    def detach_for_next_step(self) -> "DifferentiableEpisodeState":
        """copy the state and detach all tensors from the current autograd graph"""
        return DifferentiableEpisodeState(
            simulation=self.simulation.detach(),
            obs=self.obs.detach(),
            score=self.score.detach(),
            step_count=self.step_count,
            previous_action=(
                None
                if self.previous_action is None
                else self.previous_action.detach()
            ),
        )


class DifferentiableSurrogateEnv(SurrogateEnv):
    """SurrogateEnv plus a torch/autograd rollout API for SVG.

    reset()/step() remain the inherited Gym/numpy API. reset_torch()/step_torch()
    use an explicit DifferentiableEpisodeState and do not mutate the Gym episode
    state (self.state.current_params/current_obs/current_score, ...).
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
        action_smoothness_penalty_weight: float = 0.0,
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            max_steps=max_steps,
            device=device,
            reset_scale=reset_scale,
            distance_penalty_weight=distance_penalty_weight,
            action_penalty_weight=action_penalty_weight,
            action_smoothness_penalty_weight=action_smoothness_penalty_weight,
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

    def frozen_surrogate_weights(self):
        """Delegate model freezing to the simulator that owns the model."""
        return self.simulator.frozen_active_model_weights()

    def reset_torch(
        self,
        beam0: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> DifferentiableEpisodeState:
        """Start a differentiable surrogate episode without touching Gym state."""
        model_index = self.simulator.sample_model_index()
        self.simulator.set_active_model(model_index)

        beam0_t = self._prepare_beam0(beam0)
        for _ in range(MAX_TERMINAL_RESET_ATTEMPTS):
            params = clip_param_tensor_to_hw(
                self._defaults_t
                + torch.randn(N_PARAMS, device=self.device) * self._reset_std_t
            ).detach()
            simulation = self.simulator.simulate_torch(
                params=params,
                beam0=beam0_t,
                model_index=model_index,
            )
            if not bool(
                self._terminal_failure_mask(simulation.beam_states).item()
            ):
                break
        else:
            raise RuntimeError(
                "Could not sample a non-terminal differentiable initial state "
                f"after {MAX_TERMINAL_RESET_ATTEMPTS} attempts."
            )
        simulation = simulation.detach()
        score = self._score_beam_states(simulation.beam_states).detach()
        obs = self._build_obs(
            simulation.beam0, simulation.beam_states,
        ).detach()

        return DifferentiableEpisodeState(
            simulation=simulation,
            obs=obs,
            score=score,
            step_count=0,
        )

    def step_torch(
        self,
        state: DifferentiableEpisodeState,
        action: torch.Tensor,
    ) -> tuple[DifferentiableEpisodeState, torch.Tensor, bool]:
        """Apply one action and return (next_state, reward, terminated)."""

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
        params_next = clip_param_tensor_to_hw(state.simulation.params + action)
        simulation = self.simulator.simulate_torch(
            params=params_next,
            beam0=state.simulation.beam0,
            model_index=state.simulation.model_index,
        )
        terminal_mask = self._terminal_failure_mask(simulation.beam_states)
        regular_score = self._score_beam_states(simulation.beam_states)
        score_next = torch.where(
            terminal_mask,
            regular_score.new_full((), ERROR_SCORE),
            regular_score,
        )
        terminated = bool(terminal_mask.detach().item())
        if terminated:
            zero = score_next.new_zeros(())
            (
                distance_penalty,
                action_penalty,
                smoothness_penalty,
                regression_penalty,
            ) = zero, zero, zero, zero
        else:
            (
                distance_penalty,
                action_penalty,
                smoothness_penalty,
                regression_penalty,
            ) = self._differentiable_penalties(
                state=state,
                action=action,
                params_next=params_next,
                score_next=score_next,
            )
        regular_reward = (
            score_next / REWARD_SCORE_SCALE
            - distance_penalty
            - action_penalty
            - smoothness_penalty
            - regression_penalty
        )
        reward = torch.where(
            terminal_mask,
            score_next.new_full((), TERMINAL_FAILURE_REWARD),
            regular_reward,
        )
        obs_next = self._build_obs(
            simulation.beam0, simulation.beam_states,
        )

        next_state = DifferentiableEpisodeState(
            simulation=simulation,
            obs=obs_next,
            score=score_next,
            step_count=state.step_count + 1,
            previous_action=action,
        )
        return next_state, reward, terminated

    def _differentiable_penalties(
        self,
        state: DifferentiableEpisodeState,
        action: torch.Tensor,
        params_next: torch.Tensor,
        score_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Torch equivalents of BaseBeamEnv's four reward regularizers.

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

        smoothness_penalty = zero
        if (
            self.action_smoothness_penalty_weight > 0.0
            and state.previous_action is not None
        ):
            normalized_delta = torch.where(
                self._action_step_t > 0.0,
                (action - state.previous_action) / self._action_step_t,
                torch.zeros_like(action),
            )
            smoothness_penalty = (
                self.action_smoothness_penalty_weight
                * normalized_delta.square().mean()
            )

        regression_penalty = zero
        if self.score_regression_penalty_weight > 0.0:
            normalized_drop = torch.relu(state.score - score_next) / REWARD_SCORE_SCALE
            regression_penalty = self.score_regression_penalty_weight * normalized_drop

        return (
            distance_penalty,
            action_penalty,
            smoothness_penalty,
            regression_penalty,
        )

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
