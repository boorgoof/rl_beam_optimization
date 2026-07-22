"""Training/test reset-scale behavior without invoking TraceWin."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    N_STAGES,
    PARAMETERS,
    TEST_RESET_SCALE,
    TRAIN_RESET_SCALE,
    default_params,
)
from beam_optimization.env.base_beam_env import BaseBeamEnv
from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator


class _Simulator(BeamSimulator):
    def simulate(self, params):
        return BeamSimulationResult(
            params=dict(params),
            beam_states=np.zeros((N_STAGES, BEAM_STATE_DIM), dtype=np.float32),
            score_val=0.0,
            success=True,
            source="test",
        )


class _Env(BaseBeamEnv):
    def _build_simulator(self):
        return _Simulator()


class ResetScaleTests(unittest.TestCase):
    def test_environment_stores_explicit_scale_and_reports_it(self):
        env = _Env(reset_scale=TEST_RESET_SCALE)
        _, info = env.reset(seed=7)
        self.assertAlmostEqual(env.reset_scale, TEST_RESET_SCALE)
        self.assertAlmostEqual(info["reset_scale"], TEST_RESET_SCALE)
        np.testing.assert_allclose(
            env._reset_std,
            TEST_RESET_SCALE * np.asarray([p.sensitivity for p in PARAMETERS]),
            rtol=2e-6,
        )

    def test_default_environment_scale_is_training_scale(self):
        env = _Env()
        self.assertAlmostEqual(env.reset_scale, TRAIN_RESET_SCALE)

    def test_deterministic_reset_ignores_the_configured_scale(self):
        env = _Env(reset_scale=TEST_RESET_SCALE)
        _, info = env.reset(seed=7, options={"randomize_params": False})
        self.assertFalse(info["reset_randomized"])
        self.assertEqual(env._current_params, default_params())

    def test_random_reset_is_clipped_to_known_hardware_bounds(self):
        env = _Env(reset_scale=1e6)
        env.reset(seed=3)
        for spec in PARAMETERS:
            value = env._current_params[spec.key]
            if spec.hw_min is not None:
                self.assertGreaterEqual(value, spec.hw_min)
            if spec.hw_max is not None:
                self.assertLessEqual(value, spec.hw_max)

    def test_workflows_route_training_and_evaluation_scales_explicitly(self):
        root = Path(__file__).resolve().parents[1]
        train_source = (root / "scripts" / "train_policies.py").read_text()
        test_source = (root / "scripts" / "test.py").read_text()
        benchmark_source = (root / "scripts" / "benchmark.py").read_text()

        self.assertIn("reset_scale=TRAIN_RESET_SCALE", train_source)
        self.assertIn("reset_scale=TEST_RESET_SCALE", train_source)
        self.assertIn("reset_scale=TEST_RESET_SCALE", test_source)
        self.assertIn("reset_scale=TEST_RESET_SCALE", benchmark_source)


if __name__ == "__main__":
    unittest.main()
