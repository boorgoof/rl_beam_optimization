from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from beam_optimization.config.adige import ParameterSpec
from beam_optimization.config.offline_utility.refining_sensitivity import (
    _allocate_span,
    _next_span,
    compute_refined_sensitivity,
    save_report,
)
from beam_optimization.main import COMMAND_MODULES


def _parameter(
    name: str = "TEST.P1",
    key: str = "test[1]",
    *,
    sensitivity: float = 1.0,
    default: float = 0.0,
    hw_min=None,
    hw_max=None,
) -> ParameterSpec:
    return ParameterSpec(
        name,
        key,
        marker=1,
        default=default,
        sensitivity=sensitivity,
        hw_min=hw_min,
        hw_max=hw_max,
    )


class _LinearSimulator:
    def __init__(self, parameters, slope: float = 1.0, fail_at=None):
        self.parameters = tuple(parameters)
        self.slope = slope
        self.fail_at = fail_at
        self.tracewin_params = {}
        self.calls = []

    def simulate(self, params):
        seed = self.tracewin_params.get("random_seed")
        changed = []
        score = 10.0
        for parameter in self.parameters:
            value = float(params.get(parameter.key, parameter.default))
            delta = value - parameter.default
            if delta != 0.0:
                changed.append((parameter.name, delta))
                score += self.slope * delta
        self.calls.append((seed, tuple(changed)))
        if self.fail_at is not None and changed:
            if any(abs(delta) >= self.fail_at for _, delta in changed):
                return SimpleNamespace(success=False, score_val=-999.0, error="lost")
        return SimpleNamespace(success=True, score_val=score, error=None)


class _BaselineFailingSimulator(_LinearSimulator):
    def simulate(self, params):
        result = super().simulate(params)
        if not self.calls[-1][1]:
            return SimpleNamespace(success=False, score_val=-999.0, error="baseline")
        return result


class RefiningSensitivityTests(unittest.TestCase):
    def test_cli_registers_only_the_canonical_refiner(self):
        self.assertIn("refining_sensitivity", COMMAND_MODULES)
        self.assertNotIn("sensitivity2", COMMAND_MODULES)
        self.assertNotIn("sensitivity3", COMMAND_MODULES)
        self.assertNotIn("sensitivity4", COMMAND_MODULES)

    def test_current_sensitivity_converges_as_total_central_span(self):
        parameter = _parameter()
        simulator = _LinearSimulator((parameter,))
        record = compute_refined_sensitivity(
            simulator,
            parameters=(parameter,),
            seed=123,
            verbose=False,
        )[parameter.name]

        self.assertTrue(record["converged"])
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["scheme"], "central")
        self.assertAlmostEqual(record["tested_span"], 1.0)
        self.assertAlmostEqual(record["absolute_score_diff"], 1.0)
        self.assertAlmostEqual(record["sensitivity"], 1.0)

    def test_seed_is_unique_per_parameter_and_constant_within_parameter(self):
        parameters = (
            _parameter("TEST.P1", "test[1]"),
            _parameter("TEST.P2", "test[2]"),
        )
        simulator = _LinearSimulator(parameters)
        records = compute_refined_sensitivity(
            simulator,
            parameters=parameters,
            seed=9876,
            verbose=False,
        )

        seeds = [records[parameter.name]["seed"] for parameter in parameters]
        self.assertNotEqual(seeds[0], seeds[1])
        call_seeds = [call[0] for call in simulator.calls]
        self.assertEqual(call_seeds[:3], [seeds[0]] * 3)
        self.assertEqual(call_seeds[3:6], [seeds[1]] * 3)

    def test_proportional_refinement_reaches_target(self):
        parameter = _parameter(sensitivity=0.1)
        simulator = _LinearSimulator((parameter,))
        record = compute_refined_sensitivity(
            simulator,
            parameters=(parameter,),
            seed=1,
            max_iterations=6,
            verbose=False,
        )[parameter.name]

        self.assertTrue(record["converged"])
        self.assertAlmostEqual(record["sensitivity"], 1.0)
        self.assertGreater(record["iterations"], 1)

    def test_bracket_uses_geometric_midpoint(self):
        self.assertAlmostEqual(
            _next_span(
                current_span=2.0,
                score_magnitude=2.0,
                target=1.0,
                lower_span=1.0,
                upper_span=4.0,
            ),
            2.0,
        )

    def test_hardware_bound_uses_one_sided_difference(self):
        parameter = _parameter(default=0.0, hw_min=0.0, hw_max=2.0)
        self.assertEqual(_allocate_span(parameter, 1.0), (1.0, 0.0))
        simulator = _LinearSimulator((parameter,))
        record = compute_refined_sensitivity(
            simulator,
            parameters=(parameter,),
            seed=2,
            verbose=False,
        )[parameter.name]
        self.assertTrue(record["converged"])
        self.assertEqual(record["scheme"], "forward")

    def test_particle_loss_causes_backoff_then_convergence(self):
        parameter = _parameter(sensitivity=1.0)
        simulator = _LinearSimulator((parameter,), slope=4.0, fail_at=0.5)
        record = compute_refined_sensitivity(
            simulator,
            parameters=(parameter,),
            seed=3,
            max_iterations=5,
            verbose=False,
        )[parameter.name]
        self.assertTrue(record["converged"])
        self.assertAlmostEqual(record["sensitivity"], 0.25)

    def test_best_effort_uses_best_valid_measurement(self):
        parameter = _parameter(sensitivity=1.0)
        simulator = _LinearSimulator((parameter,), slope=2.0)
        record = compute_refined_sensitivity(
            simulator,
            parameters=(parameter,),
            seed=4,
            max_iterations=1,
            verbose=False,
        )[parameter.name]
        self.assertTrue(record["measured"])
        self.assertFalse(record["converged"])
        self.assertEqual(record["status"], "best_effort")
        self.assertAlmostEqual(record["sensitivity"], 0.5)

    def test_no_signal_and_baseline_failure_keep_old_sensitivity(self):
        parameter = _parameter(sensitivity=0.75)
        no_signal = compute_refined_sensitivity(
            _LinearSimulator((parameter,), slope=0.0),
            parameters=(parameter,),
            seed=5,
            max_iterations=2,
            verbose=False,
        )[parameter.name]
        failed = compute_refined_sensitivity(
            _BaselineFailingSimulator((parameter,)),
            parameters=(parameter,),
            seed=5,
            verbose=False,
        )[parameter.name]
        self.assertEqual(no_signal["sensitivity"], 0.75)
        self.assertFalse(no_signal["measured"])
        self.assertEqual(failed["sensitivity"], 0.75)
        self.assertEqual(failed["status"], "baseline_failed")

    def test_validation_and_json_report(self):
        parameter = _parameter()
        simulator = _LinearSimulator((parameter,))
        with self.assertRaises(ValueError):
            compute_refined_sensitivity(simulator, parameters=(parameter,), tolerance=0)
        with self.assertRaises(ValueError):
            compute_refined_sensitivity(simulator, parameters=(parameter,), max_iterations=0)

        record = compute_refined_sensitivity(
            simulator, parameters=(parameter,), seed=7, verbose=False
        )[parameter.name]
        with tempfile.TemporaryDirectory() as directory:
            path = save_report(
                {parameter.name: record},
                Path(directory) / "report.json",
                target_score_diff=1.0,
                tolerance=0.1,
                run_config={"master_seed": 7},
                parameters=(parameter,),
            )
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["algorithm"], "iterative_current_sensitivity_refinement")
        self.assertEqual(report["parameters"][parameter.name]["seed"], record["seed"])
        self.assertIn("relative_target_error", report["parameters"][parameter.name])


if __name__ == "__main__":
    unittest.main()
