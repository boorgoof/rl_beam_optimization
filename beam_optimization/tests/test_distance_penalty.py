"""BaseBeamEnv: optional trust-region reward penalty on param_knn_distance."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    N_STAGES,
    PARAM_KEYS,
    REWARD_SCORE_SCALE,
    TERMINAL_FAILURE_REWARD,
)
from beam_optimization.env.base_beam_env import BaseBeamEnv
from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator


def _valid_result(score: float = 20.0, npart_ratio: float = 1.0) -> BeamSimulationResult:
    beam_states = np.ones((N_STAGES, BEAM_STATE_DIM), dtype=np.float32)
    beam_states[-1, 0] = npart_ratio
    return BeamSimulationResult(
        params={},
        beam_states=beam_states,
        final_beam=None,
        score_val=score,
        success=True,
        source="test",
    )


def _physics_failure_result(error: str = "Error: All particles are lost"):
    beam_states = np.zeros((N_STAGES, BEAM_STATE_DIM), dtype=np.float32)
    beam_states[0] = 1.0
    return BeamSimulationResult(
        params={},
        beam_states=beam_states,
        final_beam=None,
        score_val=TERMINAL_FAILURE_REWARD,
        success=False,
        source="test",
        error=error,
        metadata={"physics_failure": True, "failure_beam_encoded": True},
    )


class _SequenceSimulator(BeamSimulator):
    def __init__(self, results, dataset=None):
        self.results = list(results)
        self.calls = 0
        if dataset is not None:
            self.dataset = dataset

    def simulate(self, params):
        self.calls += 1
        result = self.results.pop(0)
        result.params = dict(params)
        return result


class _Env(BaseBeamEnv):
    def __init__(self, results, *, max_steps=20, dataset=None, distance_penalty_weight=0.0):
        self.results = results
        self._dataset = dataset
        super().__init__(max_steps=max_steps, distance_penalty_weight=distance_penalty_weight)

    def _build_simulator(self):
        return _SequenceSimulator(self.results, dataset=self._dataset)


class DistancePenaltyDisabledTests(unittest.TestCase):
    def test_default_weight_leaves_reward_unchanged_even_if_distance_nonzero(self):
        env = _Env([_valid_result(), _valid_result(score=20.0, npart_ratio=1.0)])
        env.reset(options={"randomize_params": False})

        with mock.patch(
            "beam_optimization.env.base_beam_env.param_knn_distance",
            return_value=np.array([999.0]),
        ) as mocked:
            _, reward, terminated, truncated, _ = env.step(
                np.zeros(len(PARAM_KEYS), dtype=np.float32)
            )

        mocked.assert_not_called()
        self.assertAlmostEqual(reward, 20.0 / REWARD_SCORE_SCALE)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


class DistancePenaltyEnabledTests(unittest.TestCase):
    def test_penalty_subtracted_proportionally_to_distance(self):
        env = _Env(
            [_valid_result(), _valid_result(score=20.0, npart_ratio=1.0)],
            distance_penalty_weight=0.1,
        )
        env.reset(options={"randomize_params": False})

        with mock.patch(
            "beam_optimization.env.base_beam_env.param_knn_distance",
            return_value=np.array([2.0]),
        ) as mocked:
            _, reward, terminated, truncated, _ = env.step(
                np.zeros(len(PARAM_KEYS), dtype=np.float32)
            )

        mocked.assert_called_once()
        expected = 20.0 / REWARD_SCORE_SCALE - 0.1 * 2.0
        self.assertAlmostEqual(reward, expected)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_not_applied_on_terminal_failure(self):
        env = _Env(
            [_valid_result(), _physics_failure_result()],
            distance_penalty_weight=0.1,
        )
        env.reset(options={"randomize_params": False})

        with mock.patch(
            "beam_optimization.env.base_beam_env.param_knn_distance",
            return_value=np.array([2.0]),
        ) as mocked:
            _, reward, terminated, truncated, _ = env.step(
                np.zeros(len(PARAM_KEYS), dtype=np.float32)
            )

        mocked.assert_not_called()
        self.assertEqual(reward, TERMINAL_FAILURE_REWARD)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_uses_simulator_dataset_when_available(self):
        sentinel_dataset = object()
        env = _Env(
            [_valid_result(), _valid_result(score=20.0, npart_ratio=1.0)],
            dataset=sentinel_dataset,
            distance_penalty_weight=0.1,
        )
        env.reset(options={"randomize_params": False})

        with mock.patch(
            "beam_optimization.env.base_beam_env.param_knn_distance",
            return_value=np.array([1.0]),
        ) as mocked:
            env.step(np.zeros(len(PARAM_KEYS), dtype=np.float32))

        _, kwargs = mocked.call_args
        self.assertIs(kwargs["dataset"], sentinel_dataset)

    def test_falls_back_to_default_dataset_when_simulator_has_none(self):
        # _SequenceSimulator built with dataset=None never sets a `.dataset`
        # attribute at all (mirrors TraceWinSimulator, which has none).
        env = _Env(
            [_valid_result(), _valid_result(score=20.0, npart_ratio=1.0)],
            distance_penalty_weight=0.1,
        )
        env.reset(options={"randomize_params": False})
        self.assertFalse(hasattr(env.simulator, "dataset"))

        with mock.patch(
            "beam_optimization.env.base_beam_env.param_knn_distance",
            return_value=np.array([1.0]),
        ) as mocked:
            env.step(np.zeros(len(PARAM_KEYS), dtype=np.float32))

        _, kwargs = mocked.call_args
        self.assertIsNone(kwargs["dataset"])


if __name__ == "__main__":
    unittest.main()
