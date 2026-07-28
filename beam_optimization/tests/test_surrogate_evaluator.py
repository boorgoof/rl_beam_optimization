"""Complete offline surrogate-evaluation metrics and plots."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    N_OUTPUT_STAGES,
    N_PARAMS,
    score_from_matrix,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.evaluator import (
    _binary_metrics,
    _classifier_diagnostics,
    _npart_ratio_band_metrics,
    _score_metrics,
    evaluate_surrogate,
)


def _beam_values(indices: np.ndarray, *, index_scale: float = 0.01) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.float32)
    stages = np.arange(N_OUTPUT_STAGES, dtype=np.float32)
    features = np.arange(BEAM_STATE_DIM, dtype=np.float32)
    return (
        0.2
        + index_scale * indices[:, None, None]
        + 0.001 * stages[None, :, None]
        + 0.0001 * features[None, None, :]
    )


class _KnownPredictionModel(torch.nn.Module):
    def __init__(self, errors: np.ndarray, *, index_scale: float = 0.01):
        super().__init__()
        self.register_buffer("errors", torch.tensor(errors, dtype=torch.float32))
        self.index_scale = float(index_scale)

    def forward(self, stage_params, beam0):
        indices = stage_params[0][:, 0]
        features = torch.arange(
            BEAM_STATE_DIM, dtype=torch.float32, device=indices.device
        )
        outputs = []
        for stage in range(N_OUTPUT_STAGES):
            target = (
                0.2
                + self.index_scale * indices[:, None]
                + 0.001 * stage
                + 0.0001 * features[None, :]
            )
            outputs.append(target + self.errors[stage])
        return outputs


def _dataset(n_samples: int = 5, *, index_scale: float = 0.01) -> BeamDataset:
    indices = np.arange(n_samples, dtype=np.float32)
    beams = _beam_values(indices, index_scale=index_scale)
    x = np.zeros((n_samples, BEAM_STATE_DIM + N_PARAMS), dtype=np.float32)
    x[:, BEAM_STATE_DIM] = indices
    y = beams.reshape(n_samples, -1)
    scores = score_from_matrix(beams[:, -1, :]).astype(np.float32)
    dataset = BeamDataset()
    dataset.append_flat_samples(x, y, scores)
    return dataset


class SurrogateEvaluatorTests(unittest.TestCase):
    def setUp(self):
        stages = np.arange(1, N_OUTPUT_STAGES + 1, dtype=np.float32)[:, None]
        features = np.arange(1, BEAM_STATE_DIM + 1, dtype=np.float32)[None, :]
        self.errors = stages * features * 1e-4
        self.dataset = _dataset()
        self.model = _KnownPredictionModel(self.errors)

    def test_feature_stage_and_backward_compatible_metrics(self):
        result = evaluate_surrogate(
            self.model, self.dataset, batch_size=2, device="cpu"
        )

        for key in (
            "mse_all", "rmse_all", "mse_final_stage", "rmse_final_stage",
            "mse_per_stage", "rmse_per_stage",
        ):
            self.assertIn(key, result)

        expected_mse_matrix = self.errors.astype(np.float64) ** 2
        np.testing.assert_allclose(
            result["rmse_by_stage_and_feature"], self.errors, rtol=2e-5
        )
        np.testing.assert_allclose(
            result["mse_by_stage_and_feature"],
            expected_mse_matrix,
            rtol=1e-4,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result["mae_by_stage_and_feature"], self.errors, rtol=2e-5
        )
        target_std = float(np.std(np.arange(len(self.dataset)) * 0.01))
        np.testing.assert_allclose(
            result["nrmse_by_stage_and_feature"],
            self.errors / target_std,
            rtol=5e-5,
        )
        self.assertEqual(result["sample_groups"], {"n_valid": 5, "n_failures": 0})
        self.assertEqual(result["score_metrics_valid"], result["score_metrics"])
        self.assertIsNone(result["score_metrics_failures"]["mae"])
        self.assertEqual(
            result["rl_sample_groups"], {"n_valid": 5, "n_terminal": 0}
        )
        self.assertEqual(result["score_metrics_rl_valid"], result["score_metrics"])
        self.assertIsNone(result["score_metrics_rl_terminal"]["mae"])
        self.assertEqual(
            result["rl_terminal_metrics"]["regressor_only"]["n_positive"], 0
        )

        for feature_index, feature in enumerate(BEAM_STATE_FEATURES):
            metrics = result["feature_metrics"][feature]
            expected_rmse = math.sqrt(
                float(np.mean(expected_mse_matrix[:, feature_index]))
            )
            self.assertAlmostEqual(metrics["rmse_all_stages"], expected_rmse, places=7)
            self.assertAlmostEqual(
                metrics["mae_all_stages"],
                float(np.mean(self.errors[:, feature_index])),
                places=7,
            )
            self.assertAlmostEqual(
                metrics["rmse_final_stage"],
                float(self.errors[-1, feature_index]),
                places=7,
            )

    def test_final_score_metrics_match_direct_numpy_calculation(self):
        result = evaluate_surrogate(self.model, self.dataset, device="cpu")
        indices = np.arange(len(self.dataset), dtype=np.float32)
        true_final = _beam_values(indices)[:, -1, :]
        predicted_final = true_final + self.errors[-1]
        true_scores = score_from_matrix(true_final)
        predicted_scores = score_from_matrix(predicted_final)
        residuals = predicted_scores - true_scores
        expected_pearson = float(np.corrcoef(true_scores, predicted_scores)[0, 1])
        expected_r2 = float(
            1.0
            - np.sum(residuals ** 2)
            / np.sum((true_scores - np.mean(true_scores)) ** 2)
        )

        metrics = result["score_metrics"]
        self.assertAlmostEqual(metrics["mae"], float(np.mean(np.abs(residuals))), places=5)
        self.assertAlmostEqual(
            metrics["rmse"], float(np.sqrt(np.mean(residuals ** 2))), places=5
        )
        self.assertAlmostEqual(metrics["bias"], float(np.mean(residuals)), places=5)
        self.assertAlmostEqual(metrics["pearson_correlation"], expected_pearson, places=6)
        self.assertAlmostEqual(metrics["r2"], expected_r2, places=5)

    def test_constant_scores_return_null_correlation_and_r2(self):
        errors = np.zeros_like(self.errors)
        result = evaluate_surrogate(
            _KnownPredictionModel(errors, index_scale=0.0),
            _dataset(index_scale=0.0),
            device="cpu",
        )
        self.assertIsNone(result["score_metrics"]["pearson_correlation"])
        self.assertIsNone(result["score_metrics"]["r2"])

        constant_prediction = _score_metrics(
            np.arange(5, dtype=np.float64), np.ones(5, dtype=np.float64)
        )
        self.assertIsNone(constant_prediction["pearson_correlation"])
        self.assertIsNone(constant_prediction["r2"])

    def test_complete_plot_set_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_surrogate(
                self.model,
                self.dataset,
                device="cpu",
                plots_dir=temp_dir,
                plot_prefix="known_model",
            )
            self.assertEqual(
                set(result["plots"]),
                {
                    "score_scatter", "score_residuals", "rmse_heatmap",
                    "nrmse_heatmap",
                },
            )
            for path in result["plots"].values():
                target = Path(path)
                self.assertTrue(target.is_file())
                self.assertGreater(target.stat().st_size, 0)

    def test_classifier_diagnostics_include_calibration_pr_and_threshold_sweep(self):
        labels = np.array([0, 0, 1, 1], dtype=np.float64)
        proba = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
        true_scores = np.array([10.0, 20.0, -999.0, -999.0])
        predicted_scores = np.array([11.0, 19.0, 5.0, 4.0])

        result = _classifier_diagnostics(
            labels, proba, true_scores, predicted_scores, 0.5
        )

        self.assertAlmostEqual(result["brier_score"], 0.085)
        self.assertAlmostEqual(result["average_precision"], 1.0)
        self.assertEqual(len(result["calibration_bins"]), 10)
        self.assertEqual(len(result["precision_recall_curve"]), 101)
        selected = [
            row for row in result["threshold_diagnostics"]
            if np.isclose(row["threshold"], 0.5)
        ][0]
        self.assertEqual(selected["false_positives"], 0)
        self.assertEqual(selected["false_negatives"], 0)
        self.assertLess(selected["gated_score_mae"], 1.0)

    def test_rl_terminal_metrics_and_operational_bands_use_strict_ten_percent(self):
        true_ratio = np.array([0.0, 0.05, 0.10, 0.20, 0.40])
        predicted_ratio = np.array([0.02, 0.12, 0.09, 0.21, 0.08])
        true_terminal = true_ratio < 0.10
        predicted_terminal = predicted_ratio < 0.10

        terminal = _binary_metrics(true_terminal, predicted_terminal)
        self.assertEqual(
            terminal["confusion_matrix"], {"tp": 1, "fp": 2, "fn": 1, "tn": 1}
        )
        self.assertAlmostEqual(terminal["precision"], 1 / 3)
        self.assertAlmostEqual(terminal["recall"], 1 / 2)

        bands = _npart_ratio_band_metrics(
            true_ratio,
            predicted_ratio,
            classifier_proba=np.array([0.9, 0.1, 0.2, 0.8, 0.1]),
            classifier_threshold=0.5,
        )
        by_name = {band["name"]: band for band in bands}
        self.assertEqual(by_name["all_particles_lost"]["n_samples"], 1)
        self.assertEqual(by_name["rl_terminal_nonzero"]["n_samples"], 1)
        self.assertEqual(by_name["rl_valid_near_boundary"]["n_samples"], 2)
        self.assertEqual(by_name["rl_valid_above_025"]["n_samples"], 1)
        self.assertAlmostEqual(
            by_name["rl_valid_near_boundary"]["true_mean"], 0.15
        )
        self.assertEqual(
            by_name["rl_valid_near_boundary"]["pipeline_terminal_rate"], 1.0
        )

    def test_cli_defaults_to_test_and_launcher_forwards_arguments(self):
        package_root = Path(__file__).resolve().parents[1]
        evaluator_source = (
            package_root
            / "env/surrogate_env/surrogate/model/evaluator.py"
        ).read_text(encoding="utf-8")
        launcher_source = (
            package_root / "commands/evaluate_surrogate.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('def _default_test_dataset_path()', evaluator_source)
        self.assertIn('default=str(_default_test_dataset_path())', evaluator_source)
        self.assertIn('"$@"', launcher_source)


if __name__ == "__main__":
    unittest.main()
