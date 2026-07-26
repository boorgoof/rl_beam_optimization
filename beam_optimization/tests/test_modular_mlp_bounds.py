"""ModularMLP output must respect physically possible ranges."""
from __future__ import annotations

import unittest

import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    default_params,
    params_to_stage_tensors,
)
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP


_IDX = {name: i for i, name in enumerate(BEAM_STATE_FEATURES)}


class PhysicalBoundsClampTests(unittest.TestCase):
    def test_out_of_range_features_are_clamped(self):
        raw = torch.zeros((1, len(BEAM_STATE_FEATURES)))
        raw[0, _IDX["npart_ratio"]] = 1.5
        raw[0, _IDX["SizeX"]] = -3.0
        raw[0, _IDX["ex"]] = -0.2
        raw[0, _IDX["x0"]] = -7.0  # unconstrained: must survive untouched

        bounded = ModularMLP._apply_physical_bounds(raw)

        self.assertAlmostEqual(bounded[0, _IDX["npart_ratio"]].item(), 1.0)
        self.assertAlmostEqual(bounded[0, _IDX["SizeX"]].item(), 0.0)
        self.assertAlmostEqual(bounded[0, _IDX["ex"]].item(), 0.0)
        self.assertAlmostEqual(bounded[0, _IDX["x0"]].item(), -7.0)

    def test_in_range_values_are_left_untouched(self):
        raw = torch.zeros((1, len(BEAM_STATE_FEATURES)))
        raw[0, _IDX["npart_ratio"]] = 0.42
        raw[0, _IDX["SizeY"]] = 3.3
        raw[0, _IDX["y0"]] = -12.5

        bounded = ModularMLP._apply_physical_bounds(raw)

        self.assertAlmostEqual(bounded[0, _IDX["npart_ratio"]].item(), 0.42, places=5)
        self.assertAlmostEqual(bounded[0, _IDX["SizeY"]].item(), 3.3, places=5)
        self.assertAlmostEqual(bounded[0, _IDX["y0"]].item(), -12.5, places=5)

    def test_does_not_mutate_input_tensor(self):
        raw = torch.zeros((1, len(BEAM_STATE_FEATURES)))
        raw[0, _IDX["SizeX"]] = -3.0
        ModularMLP._apply_physical_bounds(raw)
        self.assertAlmostEqual(raw[0, _IDX["SizeX"]].item(), -3.0)


class ForwardOutputBoundsTests(unittest.TestCase):
    def test_every_stage_of_an_untrained_model_is_physically_bounded(self):
        model = ModularMLP()
        model.eval()
        beam0 = torch.randn(1, BEAM_STATE_DIM) * 5
        stage_tensors = params_to_stage_tensors(default_params(), device=torch.device("cpu"))

        with torch.no_grad():
            outputs = model(stage_tensors, beam0)

        for stage_output in outputs:
            npart = stage_output[:, _IDX["npart_ratio"]]
            self.assertTrue(torch.all(npart >= 0.0))
            self.assertTrue(torch.all(npart <= 1.0))
            for name in ("SizeX", "SizeY", "ex", "ey"):
                self.assertTrue(torch.all(stage_output[:, _IDX[name]] >= 0.0))


if __name__ == "__main__":
    unittest.main()
