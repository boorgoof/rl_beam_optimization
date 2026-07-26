"""SurrogateBeamSimulator.evaluate_ensemble_disagreement(): a diagnostic-only
epistemic-uncertainty signal across ensemble members."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    N_OUTPUT_STAGES,
    N_PARAMS,
    default_params,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.surrogate_simulator import (
    SurrogateBeamSimulator,
)


def _tiny_dataset(n: int = 4, seed: int = 0) -> BeamDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, BEAM_STATE_DIM + N_PARAMS)).astype(np.float32)
    Y = rng.normal(size=(n, N_OUTPUT_STAGES * BEAM_STATE_DIM)).astype(np.float32)
    scores = rng.normal(size=n).astype(np.float32)
    dataset = BeamDataset()
    dataset.append_flat_samples(X, Y, scores)
    return dataset


_IDX = {name: i for i, name in enumerate(BEAM_STATE_FEATURES)}


class _ConstantBeamModel(torch.nn.Module):
    """Deterministic stand-in for ModularMLP: same shape contract
    (stage_params, beam0) -> list[N_OUTPUT_STAGES] of (batch, BEAM_STATE_DIM),
    but a fixed, controllable output instead of a trained/random forward pass
    -- avoids the test depending on incidental untrained-network behavior."""

    def __init__(self, npart_ratio: float, size: float):
        super().__init__()
        self.npart_ratio = float(npart_ratio)
        self.size = float(size)

    def forward(self, stage_params, beam0):
        batch = beam0.shape[0]
        beam = torch.zeros((batch, BEAM_STATE_DIM))
        beam[:, _IDX["npart_ratio"]] = self.npart_ratio
        beam[:, _IDX["SizeX"]] = self.size
        beam[:, _IDX["SizeY"]] = self.size
        beam[:, _IDX["ex"]] = 0.05
        beam[:, _IDX["ey"]] = 0.05
        return [beam.clone() for _ in range(N_OUTPUT_STAGES)]


class EnsembleDisagreementTests(unittest.TestCase):
    def test_single_model_ensemble_has_zero_disagreement(self):
        dataset = _tiny_dataset()
        model = _ConstantBeamModel(npart_ratio=0.8, size=5.0)
        sim = SurrogateBeamSimulator(model, dataset, device="cpu", seed=1)

        result = sim.evaluate_ensemble_disagreement(default_params())

        self.assertEqual(result["n_models"], 1)
        self.assertEqual(result["score_std"], 0.0)
        for std in result["final_stage_std"].values():
            self.assertEqual(std, 0.0)

    def test_multi_model_ensemble_has_nonzero_disagreement(self):
        dataset = _tiny_dataset()
        ensemble = [
            _ConstantBeamModel(npart_ratio=0.6, size=5.0),
            _ConstantBeamModel(npart_ratio=0.9, size=8.0),
            _ConstantBeamModel(npart_ratio=0.75, size=6.5),
        ]
        sim = SurrogateBeamSimulator(ensemble, dataset, device="cpu", seed=1)

        result = sim.evaluate_ensemble_disagreement(default_params())

        self.assertEqual(result["n_models"], 3)
        self.assertGreater(result["score_std"], 0.0)
        self.assertGreater(result["final_stage_std"]["npart_ratio"], 0.0)
        self.assertGreater(result["final_stage_std"]["SizeX"], 0.0)
        self.assertEqual(set(result["final_stage_std"]), set(BEAM_STATE_FEATURES))

    def test_does_not_change_active_model_or_episode_state(self):
        dataset = _tiny_dataset()
        ensemble = [
            _ConstantBeamModel(npart_ratio=0.6, size=5.0),
            _ConstantBeamModel(npart_ratio=0.9, size=8.0),
        ]
        sim = SurrogateBeamSimulator(ensemble, dataset, device="cpu", seed=1)
        active_before = sim._active_model_index
        beam0_before = sim._episode_beam0.copy()

        sim.evaluate_ensemble_disagreement(default_params())

        self.assertEqual(sim._active_model_index, active_before)
        np.testing.assert_array_equal(sim._episode_beam0, beam0_before)


if __name__ == "__main__":
    unittest.main()
