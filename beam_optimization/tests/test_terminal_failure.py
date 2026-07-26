from __future__ import annotations

import unittest

import numpy as np

from beam_optimization.config.adige import (
    ALL_PARTICLES_LOST_NPART_RATIO,
    BEAM_STATE_DIM,
    ERROR_SCORE,
    MAX_TERMINAL_RESET_ATTEMPTS,
    N_STAGES,
    PARAM_KEYS,
    REWARD_SCORE_SCALE,
    RL_MIN_NPART_RATIO,
    TERMINAL_FAILURE_REWARD,
    default_params,
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
    beam_states[5] = 2.0
    return BeamSimulationResult(
        params={},
        beam_states=beam_states,
        final_beam=None,
        score_val=ERROR_SCORE,
        success=False,
        source="test",
        error=error,
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
        self.results = list(results)
        self.calls = 0

    def simulate(self, params):
        self.calls += 1
        result = self.results.pop(0)
        result.params = dict(params)
        return result


class _Env(BaseBeamEnv):
    def __init__(self, results, *, max_steps=20):
        self.results = results
        super().__init__(max_steps=max_steps)

    def _build_simulator(self):
        return _SequenceSimulator(self.results)


class TerminalFailureTests(unittest.TestCase):
    def test_all_known_tracewin_physics_failures_are_terminal(self):
        cases = (
            (
                "Error: All particles are lost",
                "all_particles_lost",
            ),
            (
                "ERROR: SYNCHRONOUS PARTICLE NEVER REACHES THE END OF THE FIELD MAP!",
                "synchronous_particle_never_reaches_end",
            ),
            (
                "Error: Part of the beam distribution never reaches the end "
                "of the field map!",
                "partial_beam_never_reaches_end",
            ),
        )
        for error, expected_reason in cases:
            with self.subTest(error=error):
                env = _Env([_valid_result(), _physics_failure_result(error)])
                env.reset(options={"randomize_params": False})
                obs, reward, terminated, truncated, info = env.step(
                    np.zeros(len(PARAM_KEYS), dtype=np.float32)
                )

                self.assertEqual(reward, TERMINAL_FAILURE_REWARD)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info["terminal_failure"])
                self.assertTrue(info["physics_failure"])
                self.assertEqual(info["failure_reason"], expected_reason)
                self.assertTrue(np.any(obs != 0.0))
                self.assertEqual(env.simulator.calls, 2)

    def test_successful_low_transmission_prediction_is_terminal(self):
        env = _Env([
            _valid_result(),
            _valid_result(score=ERROR_SCORE, npart_ratio=ALL_PARTICLES_LOST_NPART_RATIO - 0.01),
        ])
        env.reset(options={"randomize_params": False})

        _, reward, terminated, truncated, info = env.step(
            np.zeros(len(PARAM_KEYS), dtype=np.float32)
        )

        self.assertEqual(reward, TERMINAL_FAILURE_REWARD)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["physics_failure"])
        self.assertEqual(info["failure_reason"], "low_transmission")

    def test_valid_physical_score_below_rl_threshold_still_terminates_episode(self):
        # Between ALL_PARTICLES_LOST_NPART_RATIO (score() cliff) and
        # RL_MIN_NPART_RATIO (RL-only cliff): score() still returns a real,
        # non-ERROR_SCORE value -- but the RL episode ends anyway, with the
        # fixed terminal reward, not that score.
        npart_ratio = (ALL_PARTICLES_LOST_NPART_RATIO + RL_MIN_NPART_RATIO) / 2
        env = _Env([
            _valid_result(),
            _valid_result(score=42.0, npart_ratio=npart_ratio),
        ])
        env.reset(options={"randomize_params": False})

        _, reward, terminated, truncated, info = env.step(
            np.zeros(len(PARAM_KEYS), dtype=np.float32)
        )

        self.assertEqual(reward, TERMINAL_FAILURE_REWARD)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["physics_failure"])
        self.assertEqual(info["failure_reason"], "low_transmission")

    def test_classifier_style_error_score_terminates_even_above_rl_threshold(self):
        # Simulates SurrogateBeamSimulator overriding score_val to ERROR_SCORE
        # when its FailureClassifier flags a parameter set as all-particles-lost
        # (see surrogate_simulator.run_surrogate_forward), even though
        # npart_ratio itself is comfortably above RL_MIN_NPART_RATIO -- i.e.
        # the regressor's npart_ratio looks fine but the classifier disagrees.
        # The existing score_val == ERROR_SCORE fallback in
        # _terminal_failure_reason() already provides this as an OR with the
        # npart_ratio check below, with no separate wiring needed.
        env = _Env([
            _valid_result(),
            _valid_result(score=ERROR_SCORE, npart_ratio=0.5),
        ])
        env.reset(options={"randomize_params": False})

        _, reward, terminated, truncated, info = env.step(
            np.zeros(len(PARAM_KEYS), dtype=np.float32)
        )

        self.assertEqual(reward, TERMINAL_FAILURE_REWARD)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["physics_failure"])
        self.assertEqual(info["failure_reason"], "low_transmission")

    def test_exact_threshold_is_valid_and_uses_normal_reward(self):
        env = _Env([
            _valid_result(),
            _valid_result(score=20.0, npart_ratio=RL_MIN_NPART_RATIO),
        ])
        env.reset(options={"randomize_params": False})

        _, reward, terminated, truncated, info = env.step(
            np.zeros(len(PARAM_KEYS), dtype=np.float32)
        )

        self.assertAlmostEqual(reward, 20.0 / REWARD_SCORE_SCALE)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["terminal_failure"])
        self.assertIsNone(info["failure_reason"])

    def test_terminal_failure_takes_precedence_over_time_limit(self):
        env = _Env([_valid_result(), _physics_failure_result()], max_steps=1)
        env.reset(options={"randomize_params": False})

        _, _, terminated, truncated, _ = env.step(
            np.zeros(len(PARAM_KEYS), dtype=np.float32)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_physics_failure_without_beam_states_is_terminal_not_technical(self):
        failure = _physics_failure_result()
        failure.beam_states = None
        env = _Env([_valid_result(), failure])
        env.reset(options={"randomize_params": False})

        obs, reward, terminated, truncated, info = env.step(
            np.zeros(len(PARAM_KEYS), dtype=np.float32)
        )

        np.testing.assert_array_equal(obs, np.zeros_like(obs))
        self.assertEqual(reward, TERMINAL_FAILURE_REWARD)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["technical_failure"])

    def test_known_physics_message_is_terminal_even_without_metadata(self):
        failure = _physics_failure_result(
            "ERROR: SYNCHRONOUS PARTICLE NEVER REACHES THE END OF THE FIELD MAP!"
        )
        failure.beam_states = None
        failure.final_beam = None
        failure.metadata = {}
        env = _Env([_valid_result(), failure])
        env.reset(options={"randomize_params": False})

        _, reward, terminated, truncated, info = env.step(
            np.zeros(len(PARAM_KEYS), dtype=np.float32)
        )

        self.assertEqual(reward, TERMINAL_FAILURE_REWARD)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["technical_failure"])
        self.assertEqual(
            info["failure_reason"],
            "synchronous_particle_never_reaches_end",
        )

    def test_random_reset_resamples_terminal_states_in_same_call(self):
        env = _Env([
            _physics_failure_result(),
            _physics_failure_result(),
            _valid_result(),
        ])

        _, info = env.reset(seed=7)

        self.assertEqual(info["reset_attempts"], 3)
        self.assertEqual(info["rejected_terminal_resets"], 2)
        self.assertEqual(env.simulator.calls, 3)

    def test_random_reset_fails_after_configured_attempt_limit(self):
        env = _Env([
            _physics_failure_result()
            for _ in range(MAX_TERMINAL_RESET_ATTEMPTS)
        ])

        with self.assertRaisesRegex(RuntimeError, "32 Gaussian reset attempts"):
            env.reset(seed=7)
        self.assertEqual(env.simulator.calls, MAX_TERMINAL_RESET_ATTEMPTS)

    def test_explicit_or_deterministic_terminal_reset_fails_immediately(self):
        for options in (
            {"randomize_params": False},
            {"initial_params": default_params()},
        ):
            with self.subTest(options=options):
                env = _Env([_physics_failure_result()])
                with self.assertRaisesRegex(RuntimeError, "terminal physical state"):
                    env.reset(options=options)
                self.assertEqual(env.simulator.calls, 1)

    def test_technical_step_restores_state_and_truncates_neutrally(self):
        env = _Env([_valid_result(), _technical_failure_result()])
        obs_before, _ = env.reset(options={"randomize_params": False})
        params_before = env.current_params

        obs, reward, terminated, truncated, info = env.step(
            np.asarray(env.action_space.high, dtype=np.float32)
        )

        np.testing.assert_array_equal(obs, obs_before)
        self.assertEqual(env.current_params, params_before)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["technical_failure"])

    def test_technical_reset_raises_without_resampling(self):
        env = _Env([_technical_failure_result()])
        with self.assertRaisesRegex(RuntimeError, "initial episode state"):
            env.reset()
        self.assertEqual(env.simulator.calls, 1)


if __name__ == "__main__":
    unittest.main()
