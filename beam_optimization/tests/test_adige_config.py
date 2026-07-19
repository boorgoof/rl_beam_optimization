"""Structural invariants of the adige.py configuration.

These are the checks that would have caught the 16->18 parameter migration
breakage: the stage layout must stay aligned 1:1 with the output stages, and
the score variants must agree with each other.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    N_OUTPUT_STAGES,
    N_PARAMS,
    PARAMETERS,
    STAGE_MARKERS,
    STAGE_PARAM_KEYS,
    STAGE_PARAM_SIZES,
    default_params,
    params_to_stage_tensors,
    score,
    score_from_matrix,
    score_from_vec,
    score_tensor,
)


class StageLayoutTests(unittest.TestCase):
    def test_one_parameter_group_per_output_stage(self):
        self.assertEqual(len(STAGE_PARAM_SIZES), N_OUTPUT_STAGES)
        self.assertEqual(len(STAGE_PARAM_KEYS), N_OUTPUT_STAGES)
        self.assertEqual(sum(STAGE_PARAM_SIZES), N_PARAMS)

    def test_every_parameter_belongs_to_its_stage_interval(self):
        marker_by_key = {p.key: p.marker for p in PARAMETERS}
        for stage_index, keys in enumerate(STAGE_PARAM_KEYS):
            lower = STAGE_MARKERS[stage_index]
            upper = STAGE_MARKERS[stage_index + 1]
            for key in keys:
                marker = marker_by_key[key]
                self.assertTrue(
                    lower < marker <= upper,
                    f"{key} (marker {marker}) assigned to stage "
                    f"({lower}, {upper}]",
                )

    def test_params_to_stage_tensors_matches_layout(self):
        tensors = params_to_stage_tensors(default_params())
        self.assertEqual(len(tensors), N_OUTPUT_STAGES)
        for tensor, size in zip(tensors, STAGE_PARAM_SIZES):
            self.assertEqual(tuple(tensor.shape), (1, size))

    def test_modular_mlp_forward_accepts_the_layout(self):
        from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import (
            ModularMLP,
        )

        model = ModularMLP()
        outputs = model(
            params_to_stage_tensors(default_params()),
            torch.zeros(1, BEAM_STATE_DIM),
        )
        self.assertEqual(len(outputs), N_OUTPUT_STAGES)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (1, BEAM_STATE_DIM))


class ScoreConsistencyTests(unittest.TestCase):
    def _random_beams(self, n=32, seed=0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        beams = rng.normal(0.0, 1.0, size=(n, BEAM_STATE_DIM))
        beams[:, BEAM_STATE_FEATURES.index("npart_ratio")] = rng.uniform(
            -0.2, 1.2, size=n
        )
        return beams

    def test_reference_beam_scores_100(self):
        beam = {
            "npart_ratio": 1.0, "x0": 0.0, "y0": 0.0, "SizeX": 5.0,
            "SizeY": 5.0, "ex": 0.05, "ey": 0.05, "x'0": 0.0, "y'0": 0.0,
        }
        self.assertAlmostEqual(score(beam), 100.0)

    def test_all_score_variants_agree(self):
        beams = self._random_beams()
        matrix_scores = score_from_matrix(beams)
        for i, row in enumerate(beams):
            expected = score_from_vec(row.astype(np.float32))
            self.assertAlmostEqual(matrix_scores[i], expected, places=3)
        tensor_scores = score_tensor(torch.tensor(beams, dtype=torch.float64))
        np.testing.assert_allclose(
            tensor_scores.numpy(), matrix_scores, rtol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
