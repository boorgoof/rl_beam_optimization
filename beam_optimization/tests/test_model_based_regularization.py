"""Reward regularization in MBPO and differentiable SVG rollouts."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from beam_optimization.algorithms.model_based.mbpo import MBPO
from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    N_OUTPUT_STAGES,
    N_PARAMS,
    REWARD_SCORE_SCALE,
    action_step_vec,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.differentiable_surrogate_env import (
    DifferentiableBeamState,
    DifferentiableSurrogateEnv,
)


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


def _state(score: torch.Tensor, params: torch.Tensor) -> DifferentiableBeamState:
    return DifferentiableBeamState(
        beam0=torch.ones((1, BEAM_STATE_DIM)),
        params=params,
        obs=torch.ones(BEAM_STATE_DIM * 3),
        score=score,
        beam_states=[
            torch.ones((1, BEAM_STATE_DIM)) for _ in range(N_OUTPUT_STAGES)
        ],
        step_count=0,
        model_index=0,
    )


class DifferentiableRegularizationTests(unittest.TestCase):
    def _env(
        self,
        dataset: BeamDataset,
        *,
        distance: float = 0.0,
        action: float = 0.0,
        regression: float = 0.0,
    ) -> DifferentiableSurrogateEnv:
        env = DifferentiableSurrogateEnv.__new__(DifferentiableSurrogateEnv)
        env.device = torch.device("cpu")
        env.simulator = SimpleNamespace(dataset=dataset)
        env.distance_penalty_weight = distance
        env.action_penalty_weight = action
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

        distance, _, _ = env._differentiable_penalties(
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

        _, action_penalty, regression_penalty = env._differentiable_penalties(
            state,
            action,
            torch.zeros(N_PARAMS),
            score_next,
        )

        self.assertAlmostEqual(float(action_penalty.detach()), action_weight * 0.25)
        self.assertAlmostEqual(
            float(regression_penalty.detach()),
            regression_weight * 10.0 / REWARD_SCORE_SCALE,
        )
        (action_penalty + regression_penalty).backward()
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertGreater(float(action.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(score_next.grad))
        self.assertLess(float(score_next.grad), 0.0)


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
                score_regression_penalty_weight=1.0,
            )

        kwargs = env_class.call_args.kwargs
        self.assertEqual(kwargs["distance_penalty_weight"], 0.02)
        self.assertEqual(kwargs["action_penalty_weight"], 0.03)
        self.assertEqual(kwargs["score_regression_penalty_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
