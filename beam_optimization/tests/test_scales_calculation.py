"""Tests for the offline reset/action scale calibration."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from beam_optimization.config.adige import (
    ACTION_SCALE,
    DATASET_SCALE,
    MAX_STEPS,
    TEST_RESET_SCALE,
    TRAIN_RESET_SCALE,
    action_bounds,
    reset_std_vec,
    sensitivity_vec,
)
from beam_optimization.config.offline_utility.scales_calculation import (
    DEFAULT_DATASET_SCALE,
    DEFAULT_F_RESET,
    DEFAULT_K_SIGMA,
    DEFAULT_K_SIGMA_DATASET,
    compute_scales,
)


class ScaleCalculationTests(unittest.TestCase):
    def assert_budget_is_fully_used(
        self,
        scales,
        *,
        dataset_scale,
        k_sigma_dataset,
        k_sigma,
        max_steps,
    ):
        used = k_sigma * scales["train_reset_scale"] + max_steps * scales["action_scale"]
        available = k_sigma_dataset * dataset_scale
        self.assertAlmostEqual(used, available)

    def test_defaults_use_25_75_percent_budget_at_three_sigma(self):
        scales = compute_scales()
        self.assertAlmostEqual(DEFAULT_DATASET_SCALE, 0.35)
        self.assertAlmostEqual(DEFAULT_F_RESET, 0.25)
        self.assertAlmostEqual(DEFAULT_K_SIGMA_DATASET, 3.0)
        self.assertAlmostEqual(DEFAULT_K_SIGMA, 3.0)
        self.assertAlmostEqual(scales["train_reset_scale"], 0.0875)
        self.assertAlmostEqual(scales["test_reset_scale"], 0.35)
        self.assertAlmostEqual(scales["action_scale"], 0.039375)
        self.assertEqual(
            set(scales),
            {"dataset_scale", "train_reset_scale", "test_reset_scale", "action_scale"},
        )
        self.assert_budget_is_fully_used(
            scales,
            dataset_scale=DEFAULT_DATASET_SCALE,
            k_sigma_dataset=DEFAULT_K_SIGMA_DATASET,
            k_sigma=DEFAULT_K_SIGMA,
            max_steps=20,
        )

    def test_custom_inputs_use_the_complete_remaining_budget(self):
        scales = compute_scales(
            dataset_scale=0.4,
            k_sigma_dataset=2.0,
            f_reset=0.3,
            k_sigma=4.0,
            max_steps=10,
        )
        self.assertAlmostEqual(scales["train_reset_scale"], 0.06)
        self.assertAlmostEqual(scales["test_reset_scale"], 0.4)
        self.assertAlmostEqual(scales["action_scale"], 0.056)
        self.assertEqual(
            set(scales),
            {"dataset_scale", "train_reset_scale", "test_reset_scale", "action_scale"},
        )
        self.assert_budget_is_fully_used(
            scales,
            dataset_scale=0.4,
            k_sigma_dataset=2.0,
            k_sigma=4.0,
            max_steps=10,
        )

    def test_adige_uses_the_default_calculation_for_all_parameters(self):
        scales = compute_scales()
        self.assertAlmostEqual(DATASET_SCALE, scales["dataset_scale"])
        self.assertAlmostEqual(TRAIN_RESET_SCALE, scales["train_reset_scale"])
        self.assertAlmostEqual(TEST_RESET_SCALE, scales["test_reset_scale"])
        self.assertAlmostEqual(TEST_RESET_SCALE, DATASET_SCALE)
        self.assertAlmostEqual(ACTION_SCALE, scales["action_scale"])

        sensitivity = sensitivity_vec()
        low, high = action_bounds()
        expected_step = ACTION_SCALE * sensitivity
        np.testing.assert_allclose(low, -expected_step, rtol=2e-6)
        np.testing.assert_allclose(high, expected_step, rtol=2e-6)
        np.testing.assert_allclose(
            MAX_STEPS * high,
            (1.0 - DEFAULT_F_RESET)
            * DEFAULT_K_SIGMA_DATASET
            * DATASET_SCALE
            * sensitivity,
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            reset_std_vec(TRAIN_RESET_SCALE), TRAIN_RESET_SCALE * sensitivity
        )
        np.testing.assert_allclose(
            reset_std_vec(TEST_RESET_SCALE), DATASET_SCALE * sensitivity
        )

    def test_environment_notebook_uses_symbolic_reset_trajectory_plot(self):
        notebook_path = (
            Path(__file__).resolve().parents[1] / "env" / "visualize_environments.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for label in (
            "'Dataset trust region (3σ)'",
            "'Training reset (3σ)'",
            "'Test reset (1σ)'",
            "'Test reset (3σ)'",
            "'Training reset + full trajectory'",
            "formula_labels",
        ):
            self.assertIn(label, source)
        self.assertNotIn("Calculated", source)
        self.assertNotIn("Configured", source)
        self.assertNotIn("1.05 × sensitivity", source)
        self.assertNotIn("`RESET_SCALE`", source)
        self.assertIn("MIN_NPART_RATIO", source)
        self.assertIn("REWARD_SCORE_SCALE", source)
        self.assertIn("LOW_TRANSMISSION_REWARD", source)
        self.assertIn("TRAIN_RECOVERY_RESET_PROBABILITY", source)
        self.assertIn("only the selected beam states", source)
        self.assertIn(
            "{observation_dim()} float32 beam-feature values",
            source,
        )
        self.assertNotIn("+ {N_PARAMS}", source)
        self.assertIn("= {observation_dim()}'", source)
        self.assertIn("Physical beam loss does not terminate an episode", source)
        self.assertIn("technical failures truncate neutrally", source)
        self.assertNotIn("FAILURE_PENALTY", source)

if __name__ == "__main__":
    unittest.main()
