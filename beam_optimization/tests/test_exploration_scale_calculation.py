from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np

from beam_optimization.config.adige import PARAMETERS
from beam_optimization.config.offline_utility.exploration_scale_calculation import (
    DEFAULT_START_SCALE,
    calibrate_exploration_scale,
    candidate_scales,
    classify_result,
    normalized_designs,
    parameter_sets_for_scale,
)


class _AlwaysSuccessfulSimulator:
    def __init__(self):
        self.calls = 0
        self.tracewin_params = {"random_seed": 42}

    def simulate(self, params):
        self.calls += 1
        return SimpleNamespace(
            success=True,
            score_val=12.0,
            error=None,
        )


class _AlternatingSimulator(_AlwaysSuccessfulSimulator):
    def simulate(self, params):
        self.calls += 1
        return SimpleNamespace(
            success=self.calls % 2 == 1,
            score_val=12.0,
            error=None,
        )


class _SequenceSimulator(_AlwaysSuccessfulSimulator):
    def __init__(self, outcomes):
        super().__init__()
        self.outcomes = iter(outcomes)

    def simulate(self, params):
        self.calls += 1
        return SimpleNamespace(
            success=next(self.outcomes),
            score_val=12.0,
            error=None,
        )


class ExplorationScaleTests(unittest.TestCase):
    def test_candidate_grid_starts_at_default_and_descends(self):
        self.assertEqual(candidate_scales()[0], DEFAULT_START_SCALE)
        self.assertEqual(
            candidate_scales(0.5, 0.2, 0.1),
            (0.5, 0.4, 0.3, 0.2),
        )

    def test_designs_map_to_both_parameter_distributions(self):
        designs = normalized_designs(4, seed=7)
        parameter_sets = parameter_sets_for_scale(0.5, designs)
        self.assertEqual(len(parameter_sets["dataset_gaussian"]), 4)
        self.assertEqual(len(parameter_sets["bayesian_sobol"]), 4)
        self.assertEqual(
            set(parameter_sets["bayesian_sobol"][0]),
            {parameter.key for parameter in PARAMETERS},
        )

    def test_success_alone_determines_validity(self):
        low_ratio_but_successful = SimpleNamespace(
            success=True,
            final_beam={"npart_ratio": 0.1},
            beam_states=None,
        )
        valid, reason = classify_result(low_ratio_but_successful)
        self.assertTrue(valid)
        self.assertEqual(reason, "valid")

        failed = SimpleNamespace(
            success=False,
            final_beam={"npart_ratio": 0.95},
            beam_states=None,
        )
        valid, reason = classify_result(failed)
        self.assertFalse(valid)
        self.assertEqual(reason, "tracewin_failed")

    def test_report_contains_only_aggregate_success_information(self):
        simulator = _AlwaysSuccessfulSimulator()
        report = calibrate_exploration_scale(
            simulator,
            scales=(0.5,),
            n_samples=2,
            verbose=False,
        )
        distribution = report["scales"][0]["distributions"]["dataset_gaussian"]
        self.assertEqual(
            set(distribution),
            {
                "n_success",
                "n_failures",
                "n_evaluated",
                "n_total",
                "success_rate",
                "target_reachable",
                "stopped_early",
            },
        )
        self.assertNotIn("npart_ratio", str(report))

    def test_every_sample_prints_explicit_success_value(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            calibrate_exploration_scale(
                _AlternatingSimulator(),
                scales=(0.5,),
                n_samples=2,
                target_success_rate=0.5,
                sample_seed=1,
                verbose=True,
            )
        output = stream.getvalue()
        self.assertEqual(output.count("success=True"), 2)
        self.assertEqual(output.count("success=False"), 2)
        self.assertEqual(output.count("score=12"), 4)
        self.assertIn("dataset_gaussian   sample 1/2 success=True", output)
        self.assertIn("dataset_gaussian   sample 2/2 success=False", output)
        self.assertIn("Distribution 1/2: dataset_gaussian", output)
        self.assertIn("Distribution 2/2: bayesian_sobol", output)
        self.assertIn(
            "Same scale=0.5: checking bayesian_sobol "
            "(validation 2/2, not a restart).",
            output,
        )

    def test_rejected_scale_announces_the_next_lower_scale(self):
        stream = io.StringIO()
        simulator = _SequenceSimulator([False] * 4 + [True] * 64)
        with contextlib.redirect_stdout(stream):
            calibrate_exploration_scale(
                simulator,
                scales=(0.4, 0.35),
                n_samples=32,
                target_success_rate=0.9,
                sample_seed=1,
                verbose=True,
            )
        self.assertIn("Scale=0.4 rejected; trying lower scale=0.35.", stream.getvalue())

    def test_largest_successful_scale_is_selected(self):
        simulator = _AlwaysSuccessfulSimulator()
        report = calibrate_exploration_scale(
            simulator,
            scales=(0.5, 0.4),
            n_samples=4,
            target_success_rate=0.90,
            sample_seed=1,
            verbose=False,
        )
        self.assertEqual(report["selected_scale"], 0.5)
        self.assertEqual(simulator.calls, 8)
        self.assertNotIn("random_seed", simulator.tracewin_params)

    def test_29_of_32_is_accepted_for_each_distribution(self):
        outcomes = ([True] * 29 + [False] * 3) * 2
        report = calibrate_exploration_scale(
            _SequenceSimulator(outcomes),
            scales=(0.5,),
            n_samples=32,
            target_success_rate=0.90,
            sample_seed=1,
            verbose=False,
        )
        self.assertEqual(report["selected_scale"], 0.5)
        self.assertAlmostEqual(report["scales"][0]["worst_success_rate"], 29 / 32)

    def test_28_of_32_rejects_scale_even_if_other_distribution_passes(self):
        simulator = _SequenceSimulator([False] * 4 + [True] * 60)
        report = calibrate_exploration_scale(
            simulator,
            scales=(0.5,),
            n_samples=32,
            target_success_rate=0.90,
            sample_seed=1,
            verbose=False,
        )
        self.assertIsNone(report["selected_scale"])
        self.assertFalse(report["scales"][0]["accepted"])
        distributions = report["scales"][0]["distributions"]
        self.assertEqual(set(distributions), {"dataset_gaussian"})
        self.assertTrue(distributions["dataset_gaussian"]["stopped_early"])
        self.assertEqual(distributions["dataset_gaussian"]["n_evaluated"], 4)
        self.assertEqual(simulator.calls, 4)


if __name__ == "__main__":
    unittest.main()
