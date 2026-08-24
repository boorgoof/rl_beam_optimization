"""Reward regularization in MBPO and differentiable SVG rollouts."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from beam_optimization.algorithms.model_based.mbpo import MBPO
from beam_optimization.algorithms.model_based.svg import SVGAgent
from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    N_OUTPUT_STAGES,
    N_PARAMS,
    REWARD_SCORE_SCALE,
    action_step_vec,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.base_beam_env import BaseBeamEnv, EpisodeState
from beam_optimization.env.surrogate_env.differentiable_surrogate_env import (
    DifferentiableEpisodeState,
    DifferentiableSurrogateEnv,
)
from beam_optimization.env.simulation import DifferentiableBeamSimulationResult


def _dataset_with_params(param_rows: np.ndarray) -> BeamDataset:
    dataset = BeamDataset()
    dataset._X = torch.cat(
        [
            torch.zeros((len(param_rows), BEAM_STATE_DIM), dtype=torch.float32),
            torch.as_tensor(param_rows, dtype=torch.float32),
        ],
        dim=1,
    )
    return dataset


def _state(
    score: torch.Tensor,
    params: torch.Tensor,
    previous_action: torch.Tensor | None = None,
) -> DifferentiableEpisodeState:
    beam0 = torch.ones((1, BEAM_STATE_DIM))
    beam_states = [
        torch.ones((1, BEAM_STATE_DIM)) for _ in range(N_OUTPUT_STAGES)
    ]
    simulation = DifferentiableBeamSimulationResult(
        beam0=beam0,
        params=params,
        beam_states=beam_states,
        final_beam=beam_states[-1],
        score=score,
        model_index=0,
    )
    return DifferentiableEpisodeState(
        simulation=simulation,
        obs=torch.ones(BEAM_STATE_DIM * 3),
        score=score,
        step_count=0,
        previous_action=previous_action,
    )


class _PenaltyOnlyEnv(BaseBeamEnv):
    def _build_simulator(self):
        raise NotImplementedError


class DifferentiableRegularizationTests(unittest.TestCase):
    def _env(
        self,
        dataset: BeamDataset,
        *,
        distance: float = 0.0,
        action: float = 0.0,
        smoothness: float = 0.0,
        regression: float = 0.0,
    ) -> DifferentiableSurrogateEnv:
        env = DifferentiableSurrogateEnv.__new__(DifferentiableSurrogateEnv)
        env.device = torch.device("cpu")
        env.simulator = SimpleNamespace(dataset=dataset)
        env.distance_penalty_weight = distance
        env.action_penalty_weight = action
        env.action_smoothness_penalty_weight = smoothness
        env.score_regression_penalty_weight = regression
        env._action_step_t = torch.as_tensor(action_step_vec(), dtype=torch.float32)
        env._knn_reference_t = dataset.get_param_vecs()
        env._knn_std_t = torch.as_tensor(dataset.param_knn_std(), dtype=torch.float32)
        return env

    def test_knn_penalty_matches_dataset_metric_and_has_parameter_gradient(self):
        rng = np.random.default_rng(7)
        rows = rng.normal(size=(8, N_PARAMS)).astype(np.float32)
        dataset = _dataset_with_params(rows)
        weight = 0.17
        env = self._env(dataset, distance=weight)
        params = torch.tensor(
            rng.normal(size=N_PARAMS), dtype=torch.float32, requires_grad=True
        )
        state = _state(torch.tensor(20.0), torch.zeros(N_PARAMS))

        distance, _, _, _ = env._differentiable_penalties(
            state,
            torch.zeros(N_PARAMS),
            params,
            torch.tensor(20.0),
        )

        expected = weight * dataset.param_knn_distance(
            params.detach().numpy()[None, :], k=5
        )[0]
        self.assertAlmostEqual(float(distance.detach()), float(expected), places=5)
        distance.backward()
        self.assertIsNotNone(params.grad)
        self.assertTrue(torch.isfinite(params.grad).all())
        self.assertGreater(float(params.grad.abs().sum()), 0.0)

    def test_action_and_regression_penalties_match_base_formulas_and_backpropagate(self):
        dataset = _dataset_with_params(np.zeros((1, N_PARAMS), dtype=np.float32))
        action_weight = 0.04
        regression_weight = 1.5
        env = self._env(
            dataset,
            action=action_weight,
            regression=regression_weight,
        )
        action = (0.5 * env._action_step_t).detach().requires_grad_(True)
        score_next = torch.tensor(20.0, requires_grad=True)
        state = _state(torch.tensor(30.0), torch.zeros(N_PARAMS))

        _, action_penalty, smoothness_penalty, regression_penalty = (
            env._differentiable_penalties(
            state,
            action,
            torch.zeros(N_PARAMS),
            score_next,
            )
        )

        self.assertAlmostEqual(float(action_penalty.detach()), action_weight * 0.25)
        self.assertEqual(float(smoothness_penalty), 0.0)
        self.assertAlmostEqual(
            float(regression_penalty.detach()),
            regression_weight * 10.0 / REWARD_SCORE_SCALE,
        )
        (action_penalty + regression_penalty).backward()
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertGreater(float(action.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(score_next.grad))
        self.assertLess(float(score_next.grad), 0.0)

    def test_smoothness_matches_gym_formula_and_gradients_reach_both_actions(self):
        dataset = _dataset_with_params(np.zeros((1, N_PARAMS), dtype=np.float32))
        weight = 0.25
        env = self._env(dataset, smoothness=weight)
        previous_action = (-env._action_step_t).detach().requires_grad_(True)
        action = env._action_step_t.detach().requires_grad_(True)
        state = _state(
            torch.tensor(20.0),
            torch.zeros(N_PARAMS),
            previous_action=previous_action,
        )

        _, _, smoothness_penalty, _ = env._differentiable_penalties(
            state,
            action,
            torch.zeros(N_PARAMS),
            torch.tensor(20.0),
        )

        self.assertAlmostEqual(float(smoothness_penalty.detach()), 4.0 * weight)
        smoothness_penalty.backward()
        for gradient in (action.grad, previous_action.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_detach_for_next_step_detaches_previous_action(self):
        previous_action = torch.ones(N_PARAMS, requires_grad=True)
        state = _state(
            torch.tensor(20.0),
            torch.zeros(N_PARAMS),
            previous_action=previous_action,
        )

        detached = state.detach_for_next_step()

        self.assertIsNotNone(detached.previous_action)
        self.assertFalse(detached.previous_action.requires_grad)
        self.assertIsNot(detached.previous_action, previous_action)
        self.assertFalse(detached.simulation.params.requires_grad)
        self.assertFalse(detached.simulation.final_beam.requires_grad)

    def test_smoothness_value_has_numeric_parity_with_gym_environment(self):
        dataset = _dataset_with_params(np.zeros((1, N_PARAMS), dtype=np.float32))
        weight = 0.25
        torch_env = self._env(dataset, smoothness=weight)
        previous_action = -torch_env._action_step_t
        action = 0.5 * torch_env._action_step_t
        state = _state(
            torch.tensor(20.0),
            torch.zeros(N_PARAMS),
            previous_action=previous_action,
        )
        _, _, torch_penalty, _ = torch_env._differentiable_penalties(
            state,
            action,
            torch.zeros(N_PARAMS),
            torch.tensor(20.0),
        )

        gym_env = _PenaltyOnlyEnv.__new__(_PenaltyOnlyEnv)
        step = action_step_vec()
        gym_env.action_space = SimpleNamespace(low=-step, high=step)
        gym_env.action_penalty_weight = 0.0
        gym_env.action_smoothness_penalty_weight = weight
        gym_env.score_regression_penalty_weight = 0.0
        gym_env.state = EpisodeState(last_action=-step)
        _, gym_penalty, _ = gym_env._control_penalties(
            0.5 * step,
            20.0,
            20.0,
        )

        self.assertAlmostEqual(float(torch_penalty), gym_penalty, places=6)


class MBPORegularizationTests(unittest.TestCase):
    def test_penalty_weights_are_forwarded_to_synthetic_environment(self):
        agent = SimpleNamespace(device=torch.device("cpu"), replay=None)
        dataset = BeamDataset()

        with mock.patch(
            "beam_optimization.algorithms.model_based.mbpo.SurrogateEnv"
        ) as env_class:
            MBPO(
                agent=agent,
                surrogates=object(),
                dataset=dataset,
                obs_dim=3,
                act_dim=N_PARAMS,
                real_buffer_size=2,
                synth_buffer_size=2,
                distance_penalty_weight=0.02,
                action_penalty_weight=0.03,
                action_smoothness_penalty_weight=0.25,
                score_regression_penalty_weight=1.0,
            )

        kwargs = env_class.call_args.kwargs
        self.assertEqual(kwargs["distance_penalty_weight"], 0.02)
        self.assertEqual(kwargs["action_penalty_weight"], 0.03)
        self.assertEqual(kwargs["action_smoothness_penalty_weight"], 0.25)
        self.assertEqual(kwargs["score_regression_penalty_weight"], 1.0)


class SVGRegularizationTests(unittest.TestCase):
    def test_penalty_weight_is_forwarded_to_differentiable_environment(self):
        bounds = action_step_vec().astype(np.float32)
        with mock.patch(
            "beam_optimization.algorithms.model_based.svg."
            "DifferentiableSurrogateEnv"
        ) as env_class:
            SVGAgent(
                surrogate=object(),
                dataset=BeamDataset(),
                obs_dim=3,
                act_dim=N_PARAMS,
                action_bounds=(-bounds, bounds),
                param_keys=tuple(f"p{i}" for i in range(N_PARAMS)),
                default_params={f"p{i}": 0.0 for i in range(N_PARAMS)},
                hidden_dims=(4,),
                device="cpu",
                action_smoothness_penalty_weight=0.25,
            )

        self.assertEqual(
            env_class.call_args.kwargs["action_smoothness_penalty_weight"],
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
