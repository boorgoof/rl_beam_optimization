"""FailureClassifier: architecture, checkpoint I/O, label derivation, pos_weight."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from beam_optimization.config.adige import (
    ALL_PARTICLES_LOST_NPART_RATIO,
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    N_OUTPUT_STAGES,
    STAGE_PARAM_SIZES,
    default_params,
    is_all_particles_lost,
    params_to_stage_tensors,
)
from beam_optimization.env.surrogate_env.surrogate.model.failure_classifier import (
    FailureClassifier,
    compute_pos_weight,
    derive_failure_labels,
)


_NPART_IDX = BEAM_STATE_FEATURES.index("npart_ratio")


class ForwardShapeTests(unittest.TestCase):
    def test_forward_returns_one_logit_per_sample(self):
        model = FailureClassifier()
        beam0 = torch.randn(4, BEAM_STATE_DIM)
        stage_tensors = params_to_stage_tensors(default_params())
        stage_tensors = [t.expand(4, -1).contiguous() for t in stage_tensors]

        logits = model(stage_tensors, beam0)

        self.assertEqual(tuple(logits.shape), (4,))

    def test_predict_proba_is_sigmoid_of_forward_and_bounded(self):
        model = FailureClassifier()
        model.eval()
        beam0 = torch.randn(3, BEAM_STATE_DIM)
        stage_tensors = params_to_stage_tensors(default_params())
        stage_tensors = [t.expand(3, -1).contiguous() for t in stage_tensors]

        with torch.no_grad():
            logits = model(stage_tensors, beam0)
        proba = model.predict_proba(stage_tensors, beam0)

        torch.testing.assert_close(proba, torch.sigmoid(logits))
        self.assertTrue(torch.all(proba >= 0.0))
        self.assertTrue(torch.all(proba <= 1.0))


class CheckpointRoundTripTests(unittest.TestCase):
    def test_save_and_load_reproduces_predictions(self):
        model = FailureClassifier(hidden_sizes=[16, 16], dropout=0.0)
        model.eval()
        beam0 = torch.randn(2, BEAM_STATE_DIM)
        stage_tensors = params_to_stage_tensors(default_params())
        stage_tensors = [t.expand(2, -1).contiguous() for t in stage_tensors]

        with torch.no_grad():
            before = model.predict_proba(stage_tensors, beam0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classifier.pt"
            model.save(str(path))
            loaded = FailureClassifier.load(str(path))
            loaded.eval()
            with torch.no_grad():
                after = loaded.predict_proba(stage_tensors, beam0)

        torch.testing.assert_close(before, after)
        self.assertEqual(loaded.hidden_sizes, [16, 16])
        self.assertAlmostEqual(loaded.dropout, 0.0)

    def test_save_and_load_preserves_normalization_stats(self):
        norm_stats = {
            "parameter_means": [torch.zeros(size) for size in STAGE_PARAM_SIZES],
            "parameter_variances": [torch.ones(size) for size in STAGE_PARAM_SIZES],
            "beam_state_means": [torch.zeros(BEAM_STATE_DIM) for _ in range(N_OUTPUT_STAGES + 1)],
            "beam_state_variances": [torch.ones(BEAM_STATE_DIM) for _ in range(N_OUTPUT_STAGES + 1)],
        }
        model = FailureClassifier(norm_stats=norm_stats)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classifier.pt"
            model.save(str(path), extra={"normalization_metadata": norm_stats})
            loaded = FailureClassifier.load(str(path))

        self.assertTrue(loaded._has_norm)
        torch.testing.assert_close(loaded.bm_0, torch.zeros(BEAM_STATE_DIM))


class DeriveFailureLabelsTests(unittest.TestCase):
    def test_matches_is_all_particles_lost_row_by_row(self):
        rng = np.random.default_rng(0)
        n = 50
        Y = rng.normal(size=(n, N_OUTPUT_STAGES * BEAM_STATE_DIM)).astype(np.float32)
        # Force roughly half the rows below/at the cliff, half clearly above.
        final_stage = Y[:, -BEAM_STATE_DIM:]
        final_stage[: n // 2, _NPART_IDX] = ALL_PARTICLES_LOST_NPART_RATIO
        final_stage[n // 2:, _NPART_IDX] = 0.5
        Y_t = torch.as_tensor(Y)

        labels = derive_failure_labels(Y_t)

        expected = np.array([
            float(is_all_particles_lost(final_stage[i, _NPART_IDX]))
            for i in range(n)
        ], dtype=np.float32)
        np.testing.assert_array_equal(labels.numpy(), expected)

    def test_boundary_is_inclusive(self):
        Y = torch.zeros((1, N_OUTPUT_STAGES * BEAM_STATE_DIM))
        Y[0, -BEAM_STATE_DIM + _NPART_IDX] = ALL_PARTICLES_LOST_NPART_RATIO

        labels = derive_failure_labels(Y)

        self.assertEqual(labels.item(), 1.0)


class ComputePosWeightTests(unittest.TestCase):
    def test_matches_negative_over_positive_ratio(self):
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # 1 positive, 4 negative

        weight = compute_pos_weight(labels)

        self.assertAlmostEqual(weight.item(), 4.0)

    def test_falls_back_to_one_when_a_class_is_missing(self):
        all_negative = torch.zeros(5)
        all_positive = torch.ones(5)

        self.assertAlmostEqual(compute_pos_weight(all_negative).item(), 1.0)
        self.assertAlmostEqual(compute_pos_weight(all_positive).item(), 1.0)


if __name__ == "__main__":
    unittest.main()
