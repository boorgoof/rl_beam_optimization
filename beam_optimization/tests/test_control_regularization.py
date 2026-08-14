"""Training reward regularization for smooth, non-oscillating policies."""
from __future__ import annotations

import unittest

import numpy as np

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    N_STAGES,
    REWARD_SCORE_SCALE,
)
from beam_optimization.env.base_beam_env import BaseBeamEnv
from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator


def _valid_result(score: float) -> BeamSimulationResult:
    beam_states = np.ones((N_STAGES, BEAM_STATE_DIM), dtype=np.float32)
    return BeamSimulationResult(
        params={},
        beam_states=beam_states,
        final_beam=None,
        score_val=score,
        success=True,
        source="test",
    )


class _SequenceSimulator(BeamSimulator):
    def __init__(self, scores):
        self.results = [_valid_result(score) for score in scores]

    def simulate(self, params):
        result = self.results.pop(0)
        result.params = dict(params)
        return result


class _Env(BaseBeamEnv):
    def __init__(
        self,
        scores,
        *,
        action_penalty_weight=0.0,
        action_smoothness_penalty_weight=0.0,
        score_regression_penalty_weight=0.0,
    ):
        self.scores = scores
        super().__init__(
            action_penalty_weight=action_penalty_weight,
            action_smoothness_penalty_weight=action_smoothness_penalty_weight,
            score_regression_penalty_weight=score_regression_penalty_weight,
        )

    def _build_simulator(self):
        return _SequenceSimulator(self.scores)


class ControlRegularizationTests(unittest.TestCase):
    def test_disabled_regularization_preserves_original_reward(self):
        env = _Env([30.0, 20.0])
        env.reset(options={"randomize_params": False})

        _, reward, _, _, info = env.step(env.action_space.high)

        self.assertAlmostEqual(reward, 20.0 / REWARD_SCORE_SCALE)
        self.assertEqual(info["action_penalty"], 0.0)
        self.assertEqual(info["action_smoothness_penalty"], 0.0)
        self.assertEqual(info["score_regression_penalty"], 0.0)

    def test_maximum_action_pays_one_weight_independent_of_parameter_units(self):
        weight = 0.03
        env = _Env([20.0, 20.0], action_penalty_weight=weight)
        env.reset(options={"randomize_params": False})

        _, reward, _, _, info = env.step(env.action_space.high)

        self.assertAlmostEqual(info["action_penalty"], weight)
        self.assertAlmostEqual(reward, 20.0 / REWARD_SCORE_SCALE - weight)

    def test_quadratic_action_penalty_scales_with_squared_fraction(self):
        weight = 0.04
        env = _Env([20.0, 20.0], action_penalty_weight=weight)
        env.reset(options={"randomize_params": False})

        _, _, _, _, info = env.step(0.5 * env.action_space.high)

        self.assertAlmostEqual(info["action_penalty"], weight * 0.25)

    def test_score_drop_receives_extra_normalized_penalty(self):
        weight = 1.5
        env = _Env(
            [30.0, 20.0],
            score_regression_penalty_weight=weight,
        )
        env.reset(options={"randomize_params": False})

        _, reward, _, _, info = env.step(np.zeros_like(env.action_space.high))

        expected_penalty = weight * (30.0 - 20.0) / REWARD_SCORE_SCALE
        self.assertAlmostEqual(info["score_regression_penalty"], expected_penalty)
        self.assertAlmostEqual(reward, 20.0 / REWARD_SCORE_SCALE - expected_penalty)

    def test_first_step_and_repeated_action_have_no_smoothness_penalty(self):
        env = _Env(
            [20.0, 20.0, 20.0],
            action_smoothness_penalty_weight=0.25,
        )
        env.reset(options={"randomize_params": False})
        action = np.asarray(env.action_space.high, dtype=np.float32)

        _, _, _, _, first_info = env.step(action)
        _, _, _, _, second_info = env.step(action)

        self.assertEqual(first_info["action_smoothness_penalty"], 0.0)
        self.assertEqual(second_info["action_smoothness_penalty"], 0.0)

    def test_opposite_bound_actions_pay_four_times_the_weight(self):
        weight = 0.25
        env = _Env(
            [20.0, 20.0, 20.0],
            action_smoothness_penalty_weight=weight,
        )
        env.reset(options={"randomize_params": False})
        env.step(env.action_space.high)

        _, reward, _, _, info = env.step(env.action_space.low)

        self.assertAlmostEqual(info["action_smoothness_penalty"], 4.0 * weight)
        self.assertAlmostEqual(reward, 20.0 / REWARD_SCORE_SCALE - 4.0 * weight)

    def test_actions_are_clipped_before_smoothness_is_computed(self):
        weight = 0.2
        env = _Env(
            [20.0, 20.0, 20.0],
            action_smoothness_penalty_weight=weight,
        )
        env.reset(options={"randomize_params": False})
        env.step(10.0 * env.action_space.high)

        _, _, _, _, info = env.step(10.0 * env.action_space.low)

        self.assertAlmostEqual(info["action_smoothness_penalty"], 4.0 * weight)

    def test_reset_forgets_previous_action(self):
        env = _Env(
            [20.0, 20.0, 20.0, 20.0],
            action_smoothness_penalty_weight=0.25,
        )
        env.reset(options={"randomize_params": False})
        env.step(env.action_space.high)
        env.reset(options={"randomize_params": False})

        _, _, _, _, info = env.step(env.action_space.low)

        self.assertEqual(info["action_smoothness_penalty"], 0.0)

    def test_score_improvement_has_no_regression_penalty(self):
        env = _Env(
            [20.0, 30.0],
            score_regression_penalty_weight=2.0,
        )
        env.reset(options={"randomize_params": False})

        _, reward, _, _, info = env.step(np.zeros_like(env.action_space.high))

        self.assertEqual(info["score_regression_penalty"], 0.0)
        self.assertAlmostEqual(reward, 30.0 / REWARD_SCORE_SCALE)

    def test_reset_reinitializes_best_episode_state(self):
        env = _Env([10.0, 30.0, 20.0])
        env.reset(options={"randomize_params": False})
        env.step(np.zeros_like(env.action_space.high))

        self.assertEqual(env.state.best_score, 30.0)
        self.assertEqual(env.state.best_step, 1)

        env.reset(options={"randomize_params": False})

        self.assertEqual(env.state.best_score, 20.0)
        self.assertEqual(env.state.best_step, 0)
        self.assertEqual(env.state.best_params, env.current_params)

    def test_negative_or_nonfinite_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            _Env([20.0], action_penalty_weight=-0.1)
        with self.assertRaises(ValueError):
            _Env([20.0], score_regression_penalty_weight=float("nan"))
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _Env([20.0], action_smoothness_penalty_weight=value)


if __name__ == "__main__":
    unittest.main()
