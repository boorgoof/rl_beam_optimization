"""SurrogateBeamSimulator: optional FailureClassifier gating of simulate()'s
score_val near the all-particles-lost cliff (surrogate_simulator.py)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    ERROR_SCORE,
    N_OUTPUT_STAGES,
    N_PARAMS,
    default_params,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.surrogate_simulator import (
    SurrogateBeamSimulator,
)


_IDX = {name: i for i, name in enumerate(BEAM_STATE_FEATURES)}


def _tiny_dataset(n: int = 4, seed: int = 0) -> BeamDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, BEAM_STATE_DIM + N_PARAMS)).astype(np.float32)
    Y = rng.normal(size=(n, N_OUTPUT_STAGES * BEAM_STATE_DIM)).astype(np.float32)
    scores = rng.normal(size=n).astype(np.float32)
    dataset = BeamDataset()
    dataset.append_flat_samples(X, Y, scores)
    return dataset


class _ConstantBeamModel(torch.nn.Module):
    """Deterministic ModularMLP stand-in with a comfortably non-failing
    prediction (npart_ratio=0.8), so any ERROR_SCORE observed in these tests
    can only have come from the classifier gate, not from the regressor."""

    def forward(self, stage_params, beam0):
        batch = beam0.shape[0]
        beam = torch.zeros((batch, BEAM_STATE_DIM))
        beam[:, _IDX["npart_ratio"]] = 0.8
        beam[:, _IDX["SizeX"]] = 5.0
        beam[:, _IDX["SizeY"]] = 5.0
        return [beam.clone() for _ in range(N_OUTPUT_STAGES)]


class _ConstantClassifier(torch.nn.Module):
    """Deterministic FailureClassifier stand-in: always predicts a fixed
    failure probability, regardless of input."""

    def __init__(self, proba: float):
        super().__init__()
        self.proba = float(proba)

    def predict_proba(self, stage_params, beam_state_0):
        return torch.full((beam_state_0.shape[0],), self.proba)


class SimulatorClassifierGatingTests(unittest.TestCase):
    def test_classifier_always_failing_overrides_score_to_error(self):
        dataset = _tiny_dataset()
        sim = SurrogateBeamSimulator(
            _ConstantBeamModel(), dataset, device="cpu", seed=1,
            classifier=_ConstantClassifier(proba=0.9),
        )

        result = sim.simulate(default_params())

        self.assertEqual(result.score_val, ERROR_SCORE)
        self.assertTrue(result.metadata["classifier_flagged_failure"])

    def test_classifier_always_ok_leaves_regressor_score_untouched(self):
        dataset = _tiny_dataset()
        sim = SurrogateBeamSimulator(
            _ConstantBeamModel(), dataset, device="cpu", seed=1,
            classifier=_ConstantClassifier(proba=0.1),
        )

        result = sim.simulate(default_params())

        self.assertNotEqual(result.score_val, ERROR_SCORE)
        self.assertFalse(result.metadata["classifier_flagged_failure"])

    def test_no_classifier_matches_pre_existing_behavior(self):
        dataset = _tiny_dataset()
        sim = SurrogateBeamSimulator(_ConstantBeamModel(), dataset, device="cpu", seed=1)

        result = sim.simulate(default_params())

        self.assertNotEqual(result.score_val, ERROR_SCORE)
        self.assertNotIn("classifier_flagged_failure", result.metadata)

    def test_threshold_is_configurable(self):
        dataset = _tiny_dataset()
        sim = SurrogateBeamSimulator(
            _ConstantBeamModel(), dataset, device="cpu", seed=1,
            classifier=_ConstantClassifier(proba=0.4),
            classifier_threshold=0.3,
        )

        result = sim.simulate(default_params())

        self.assertEqual(result.score_val, ERROR_SCORE)
        self.assertTrue(result.metadata["classifier_flagged_failure"])


if __name__ == "__main__":
    unittest.main()
