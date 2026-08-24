"""Shared simulator forward and the explicit differentiable SVG state."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    N_OUTPUT_STAGES,
    N_PARAMS,
    REWARD_SCORE_SCALE,
    default_params,
    params_to_vec,
    score_tensor,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env import DifferentiableSurrogateEnv
from beam_optimization.env.surrogate_env.surrogate.surrogate_simulator import (
    SurrogateBeamSimulator,
)


_NPART_INDEX = BEAM_STATE_FEATURES.index("npart_ratio")


def _dataset() -> BeamDataset:
    x = np.zeros((4, BEAM_STATE_DIM + N_PARAMS), dtype=np.float32)
    x[:, _NPART_INDEX] = 0.9
    y = np.zeros((4, N_OUTPUT_STAGES * BEAM_STATE_DIM), dtype=np.float32)
    scores = np.zeros(4, dtype=np.float32)
    dataset = BeamDataset()
    dataset.append_flat_samples(x, y, scores)
    return dataset


class _LinearBeamModel(torch.nn.Module):
    """Small deterministic model whose output depends on every parameter."""

    def __init__(self, offset: float = 0.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.offset = float(offset)

    def forward(self, stage_params, beam0):
        flat_params = torch.cat(stage_params, dim=1)
        influence = flat_params.sum(dim=1, keepdim=True) * self.scale * 1e-4
        outputs = []
        for stage_index in range(N_OUTPUT_STAGES):
            stage = beam0 + influence * (stage_index + 1) + self.offset
            columns = list(stage.unbind(dim=1))
            columns[_NPART_INDEX] = torch.sigmoid(columns[_NPART_INDEX])
            outputs.append(torch.stack(columns, dim=1))
        return outputs


class DifferentiableSimulatorTests(unittest.TestCase):
    def test_simulate_and_simulate_torch_have_numeric_parity(self):
        simulator = SurrogateBeamSimulator(
            _LinearBeamModel(), _dataset(), device="cpu", seed=3,
        )
        params = default_params()
        beam0 = simulator._episode_beam0.copy()

        normal = simulator.simulate(params)
        differentiable = simulator.simulate_torch(
            torch.tensor(params_to_vec(params)),
            torch.tensor(beam0),
        )
        differentiable_stages = np.asarray(
            [
                differentiable.beam0.squeeze(0).detach().numpy(),
                *[
                    stage.squeeze(0).detach().numpy()
                    for stage in differentiable.beam_states
                ],
            ],
            dtype=np.float32,
        )

        self.assertTrue(normal.success)
        np.testing.assert_allclose(
            normal.beam_states, differentiable_stages, rtol=1e-6, atol=1e-6,
        )
        self.assertAlmostEqual(
            normal.score, float(differentiable.score.detach()), places=5,
        )

    def test_score_and_beam_states_backpropagate_to_flat_params(self):
        simulator = SurrogateBeamSimulator(
            _LinearBeamModel(), _dataset(), device="cpu", seed=3,
        )
        params = torch.tensor(
            params_to_vec(default_params()), requires_grad=True,
        )
        result = simulator.simulate_torch(
            params, torch.tensor(simulator._episode_beam0),
        )

        (result.score.sum() + result.beam_states[0].sum()).backward()

        self.assertIsNotNone(params.grad)
        self.assertTrue(torch.isfinite(params.grad).all())
        self.assertGreater(float(params.grad.abs().sum()), 0.0)

    def test_requested_ensemble_member_is_used_and_recorded(self):
        simulator = SurrogateBeamSimulator(
            [_LinearBeamModel(0.0), _LinearBeamModel(0.2)],
            _dataset(),
            device="cpu",
            seed=3,
        )
        params = torch.tensor(params_to_vec(default_params()))
        beam0 = torch.tensor(simulator._episode_beam0)

        first = simulator.simulate_torch(params, beam0, model_index=0)
        second = simulator.simulate_torch(params, beam0, model_index=1)

        self.assertEqual(first.model_index, 0)
        self.assertEqual(second.model_index, 1)
        self.assertFalse(torch.allclose(first.final_beam, second.final_beam))

    def test_freezing_model_weights_preserves_parameter_gradients(self):
        model = _LinearBeamModel()
        simulator = SurrogateBeamSimulator(model, _dataset(), device="cpu", seed=3)
        params = torch.tensor(
            params_to_vec(default_params()), requires_grad=True,
        )
        with simulator.frozen_active_model_weights():
            result = simulator.simulate_torch(
                params, torch.tensor(simulator._episode_beam0),
            )
            result.score.sum().backward()
            self.assertFalse(model.scale.requires_grad)
        self.assertTrue(model.scale.requires_grad)
        self.assertIsNotNone(params.grad)
        self.assertIsNone(model.scale.grad)


class DifferentiableEnvironmentStateTests(unittest.TestCase):
    def _env(self, stage_weights=None) -> DifferentiableSurrogateEnv:
        return DifferentiableSurrogateEnv(
            model=_LinearBeamModel(),
            dataset=_dataset(),
            max_steps=3,
            device="cpu",
            stage_weights=stage_weights,
        )

    def test_torch_api_does_not_mutate_inherited_gym_state(self):
        env = self._env()
        gym_state = env.state
        gym_params = dict(gym_state.current_params)
        gym_obs = gym_state.current_obs.copy()

        state = env.reset_torch()
        env.step_torch(state, torch.zeros(N_PARAMS))

        self.assertIs(env.state, gym_state)
        self.assertEqual(env.state.current_params, gym_params)
        np.testing.assert_array_equal(env.state.current_obs, gym_obs)

    def test_two_steps_keep_full_bptt_graph(self):
        env = self._env()
        state = env.reset_torch()
        action1 = torch.zeros(N_PARAMS, requires_grad=True)
        state, _, terminated = env.step_torch(state, action1)
        self.assertFalse(terminated)
        action2 = torch.zeros(N_PARAMS, requires_grad=True)
        state, reward, terminated = env.step_torch(state, action2)
        self.assertFalse(terminated)

        reward.backward()

        for action in (action1, action2):
            self.assertIsNotNone(action.grad)
            self.assertGreater(float(action.grad.abs().sum()), 0.0)

    def test_stage_weighted_score_and_reward_keep_the_existing_formula(self):
        raw_weights = list(range(1, N_OUTPUT_STAGES + 1))
        env = self._env(stage_weights=raw_weights)
        state = env.reset_torch()

        next_state, reward, terminated = env.step_torch(
            state, torch.zeros(N_PARAMS),
        )

        self.assertFalse(terminated)
        stage_scores = torch.stack(
            [score_tensor(stage) for stage in next_state.simulation.beam_states]
        )
        weights = torch.tensor(raw_weights, dtype=torch.float32)
        weights = weights / weights.sum()
        expected_score = (stage_scores * weights.view(-1, 1)).sum(dim=0)
        torch.testing.assert_close(next_state.score, expected_score)
        torch.testing.assert_close(reward, expected_score / REWARD_SCORE_SCALE)


if __name__ == "__main__":
    unittest.main()
