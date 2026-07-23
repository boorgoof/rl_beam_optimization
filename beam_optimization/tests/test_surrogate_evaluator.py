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
                {"score_scatter", "score_residuals", "rmse_heatmap"},
            )
            for path in result["plots"].values():
                target = Path(path)
                self.assertTrue(target.is_file())
                self.assertGreater(target.stat().st_size, 0)

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
