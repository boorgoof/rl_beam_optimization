from __future__ import annotations

import unittest

import numpy as np

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    ERROR_SCORE,
    LOW_TRANSMISSION_REWARD,
    N_STAGES,
    PARAM_KEYS,
    REWARD_SCORE_SCALE,
)
from beam_optimization.env.base_beam_env import BaseBeamEnv
from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator


def _valid_result(score: float = 20.0) -> BeamSimulationResult:
    beam_states = np.ones((N_STAGES, BEAM_STATE_DIM), dtype=np.float32)
    return BeamSimulationResult(
        params={},
        beam_states=beam_states,
        final_beam=None,
        score_val=score,
        success=True,
        source="test",
    )


def _physics_failure_result() -> BeamSimulationResult:
    beam_states = np.zeros((N_STAGES, BEAM_STATE_DIM), dtype=np.float32)
    beam_states[0] = 1.0
    beam_states[5] = 2.0
    return BeamSimulationResult(
        params={},
        beam_states=beam_states,
        final_beam=None,
        score_val=ERROR_SCORE,
        success=False,
        source="test",
        error="Error: All particles are lost",
        metadata={"physics_failure": True, "failure_beam_encoded": True},
    )


def _technical_failure_result() -> BeamSimulationResult:
    return BeamSimulationResult(
        params={},
        beam_states=None,
        final_beam=None,
        score_val=ERROR_SCORE,
        success=False,
        source="test",
        error="Qt platform plugin failed",
    )


class _SequenceSimulator(BeamSimulator):
    def __init__(self, results):
        self._results = list(results)

    def simulate(self, params):
        return self._results.pop(0)


class _Env(BaseBeamEnv):
    def __init__(self, results, *, max_steps=20):
        self._results = results
        super().__init__(max_steps=max_steps)

    def _build_simulator(self):
        return _SequenceSimulator(self._results)


class RecoveryRewardTests(unittest.TestCase):
    def test_failure_recovery_cycle_cannot_farm_positive_reward(self):
        env = _Env([
            _valid_result(),
            _physics_failure_result(),
            _physics_failure_result(),
            _valid_result(),
        ])
        env.reset(options={"randomize_params": False})
        action = np.zeros(len(PARAM_KEYS), dtype=np.float32)

        failed_obs, reward_1, terminated_1, truncated_1, info_1 = env.step(action)
        _, reward_2, terminated_2, truncated_2, _ = env.step(action)
        _, recovery_reward, terminated_3, truncated_3, _ = env.step(action)

        self.assertAlmostEqual(reward_1, LOW_TRANSMISSION_REWARD)
        self.assertAlmostEqual(reward_2, LOW_TRANSMISSION_REWARD)
        self.assertAlmostEqual(recovery_reward, 20.0 / REWARD_SCORE_SCALE)
        self.assertLess(reward_1 + reward_2 + recovery_reward, 0.0)
        self.assertFalse(any((terminated_1, terminated_2, terminated_3)))
        self.assertFalse(any((truncated_1, truncated_2, truncated_3)))
        self.assertTrue(info_1["physics_failure"])
        self.assertTrue(np.any(failed_obs != 0.0))

    def test_physics_failure_only_truncates_at_max_steps(self):
        env = _Env(
            [_physics_failure_result(), _physics_failure_result(), _physics_failure_result()],
            max_steps=2,
        )
        env.reset(options={"randomize_params": False})
        action = np.zeros(len(PARAM_KEYS), dtype=np.float32)

        _, _, terminated_1, truncated_1, _ = env.step(action)
        _, _, terminated_2, truncated_2, _ = env.step(action)

        self.assertFalse(terminated_1)
        self.assertFalse(truncated_1)
        self.assertFalse(terminated_2)
        self.assertTrue(truncated_2)

    def test_technical_step_restores_state_and_truncates_neutrally(self):
        env = _Env([_valid_result(), _technical_failure_result()])
        obs_before, _ = env.reset(options={"randomize_params": False})
        params_before = env.current_params
        action = np.asarray(env.action_space.high, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        np.testing.assert_array_equal(obs, obs_before)
        self.assertEqual(env.current_params, params_before)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["technical_failure"])

    def test_technical_reset_raises_instead_of_creating_fake_state(self):
        env = _Env([_technical_failure_result()])
        with self.assertRaisesRegex(RuntimeError, "initial episode state"):
            env.reset(options={"randomize_params": False})

    def test_render_extracts_selected_beam_observation(self):
        env = _Env([_valid_result()])
        obs, _ = env.reset(options={"randomize_params": False})
        frame = env._obs_to_stage_frame(obs)
        self.assertEqual(len(frame), 3)
        self.assertEqual(list(frame.columns[1:]), list(
            ("npart_ratio", "x0", "y0", "SizeX", "SizeY", "ex", "ey", "x'0", "y'0")
        ))


if __name__ == "__main__":
    unittest.main()
