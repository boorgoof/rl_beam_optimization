from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from beam_optimization.config.offline_utility import fail_scale_calculation as module


def _result(success: bool, error=None, score=1.0):
    return SimpleNamespace(success=success, error=error, score_val=score)


class _SequenceSimulator:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = 0
        self.tracewin_params = {"random_seed": 123}

    def simulate(self, params):
        self.calls += 1
        return next(self.results)


class FailScaleCalculationTests(unittest.TestCase):
    def test_all_three_physics_messages_are_case_insensitive(self):
        messages = [
            "ERROR: ALL PARTICLES ARE LOST",
            "Synchronous Particle Never Reaches The End Of The Field Map!",
            "Part Of The Beam Distribution Never Reaches The End Of The Field Map!",
        ]
        for message in messages:
            classification, original = module.classify_result(_result(False, message))
            self.assertEqual(classification, "physics_failure")
            self.assertEqual(original, message)

    def test_unknown_failure_aborts(self):
        with self.assertRaisesRegex(RuntimeError, "calibration aborted"):
            module.classify_result(_result(False, "SSH connection failed"))

    def test_29_of_32_is_accepted_and_28_is_rejected(self):
        failure = _result(False, "Error: All particles are lost", score=-999)
        success = _result(True)
        params = [{} for _ in range(32)]

        accepted_sim = _SequenceSimulator([failure] * 29 + [success] * 3)
        accepted = module.evaluate_scale(
            accepted_sim, params, scale=1.0, target_failure_rate=0.9, verbose=False
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["n_physics_failures"], 29)
        self.assertEqual(accepted_sim.calls, 29)

        rejected_sim = _SequenceSimulator([failure] * 28 + [success] * 4)
        rejected = module.evaluate_scale(
            rejected_sim, params, scale=1.0, target_failure_rate=0.9, verbose=False
        )
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["n_physics_failures"], 28)
        self.assertEqual(rejected_sim.calls, 32)

    def test_expansion_and_bisection_find_smallest_tested_upper_bound(self):
        scales = []

        def fake_evaluate(simulator, params, *, scale, target_failure_rate, verbose):
            scales.append(scale)
            return {
                "scale": scale,
                "accepted": scale >= 0.8,
                "required_failures": 1,
                "n_physics_failures": int(scale >= 0.8),
                "n_non_failures": int(scale < 0.8),
                "n_evaluated": 1,
                "n_total": 1,
                "physical_failure_rate_lower_bound": float(scale >= 0.8),
                "stopped_early": False,
                "samples": [],
            }

        with patch.object(module, "evaluate_scale", side_effect=fake_evaluate):
            report = module.calibrate_fail_scale(
                object(),
                start_scale=0.5,
                max_scale=2.0,
                expansion_factor=2.0,
                bisection_iterations=2,
                n_samples=1,
                sample_seed=1,
                verbose=False,
            )
        self.assertEqual(scales, [0.5, 1.0, 0.75, 0.875])
        self.assertEqual(report["selected_scale"], 0.875)
        self.assertEqual(report["bracket"], {
            "lower_below_target": 0.75,
            "upper_reaches_target": 0.875,
        })

    def test_no_threshold_leaves_scale_unset(self):
        def always_rejected(simulator, params, *, scale, target_failure_rate, verbose):
            return {
                "scale": scale, "accepted": False, "required_failures": 1,
                "n_physics_failures": 0, "n_non_failures": 1,
                "n_evaluated": 1, "n_total": 1,
                "physical_failure_rate_lower_bound": 0.0,
                "stopped_early": False, "samples": [],
            }
        with patch.object(module, "evaluate_scale", side_effect=always_rejected):
            report = module.calibrate_fail_scale(
                object(), start_scale=0.5, max_scale=1.0,
                expansion_factor=2.0, n_samples=1, verbose=False,
            )
        self.assertIsNone(report["selected_scale"])
        self.assertEqual(report["status"], "no_failure_scale_found")

    def test_config_update_is_explicit_and_atomic_result_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adige.py"
            path.write_text("BEFORE = 1\nALL_PARTICLE_LOST_SCALE = None\nAFTER = 2\n")
            module.update_adige_fail_scale(1.2345, path)
            text = path.read_text()
        self.assertIn("ALL_PARTICLE_LOST_SCALE: float = 1.234500000000000e+00", text)
        self.assertIn("BEFORE = 1", text)
        self.assertIn("AFTER = 2", text)


if __name__ == "__main__":
    unittest.main()
