from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from beam_optimization.config.adige import BEAM_STATE_FEATURES
from beam_optimization.env.tracewin_env.tracewin_env import TraceWinEnv


class TraceWinDistributionRenderTests(unittest.TestCase):
    def test_default_render_matches_distribution_notebook_ranges(self):
        beam_state = {
            feature: (1.0 if feature == "npart_ratio" else 0.0)
            for feature in BEAM_STATE_FEATURES
        }
        distribution = {
            key: np.asarray([0.0, 1e-3], dtype=float)
            for key in ("x", "y", "xp", "yp")
        }
        sentinel = object()

        with tempfile.TemporaryDirectory() as temporary:
            calc_dir = Path(temporary)
            (calc_dir / "part_dtl1.dst").touch()
            env = TraceWinEnv.__new__(TraceWinEnv)
            env.simulator = SimpleNamespace(calc_dir=str(calc_dir))
            env._step_count = 0
            env._current_result = SimpleNamespace(
                success=True,
                final_beam=beam_state,
                score_val=12.5,
                metadata={"sim_count": 1},
            )

            with (
                mock.patch(
                    "beam_optimization.env.tracewin_env.tracewin.visualization."
                    "tracewin_distribution_from_dst",
                    return_value=distribution,
                ) as load_distribution,
                mock.patch(
                    "beam_optimization.env.tracewin_env.tracewin.visualization."
                    "plot_tracewin_distribution_with_state",
                    return_value=sentinel,
                ) as plot_distribution,
            ):
                result = env.render_final_beam_distribution()

        self.assertIs(result, sentinel)
        load_distribution.assert_called_once_with(
            calc_dir / "part_dtl1.dst",
            max_particles=None,
        )
        kwargs = plot_distribution.call_args.kwargs
        self.assertEqual(kwargs["bins"], 200)
        self.assertEqual(kwargs["xy_range_mm"], 20.0)
        self.assertEqual(kwargs["angle_range_mrad"], 40.0)
        self.assertEqual(kwargs["figsize"], (22, 8.5))
        self.assertEqual(
            plot_distribution.call_args.args[2],
            12.5,
        )
        self.assertEqual(kwargs["state_source"], "environment step 0")

    def test_explicit_legacy_axis_range_controls_both_axes(self):
        beam_state = {
            feature: (1.0 if feature == "npart_ratio" else 0.0)
            for feature in BEAM_STATE_FEATURES
        }
        distribution = {
            key: np.asarray([0.0], dtype=float)
            for key in ("x", "y", "xp", "yp")
        }

        with tempfile.TemporaryDirectory() as temporary:
            calc_dir = Path(temporary)
            (calc_dir / "part_dtl1.dst").touch()
            env = TraceWinEnv.__new__(TraceWinEnv)
            env.simulator = SimpleNamespace(calc_dir=str(calc_dir))
            env._step_count = 0
            env._current_result = SimpleNamespace(
                success=True,
                final_beam=beam_state,
                score_val=0.0,
                metadata={"sim_count": 1},
            )

            with (
                mock.patch(
                    "beam_optimization.env.tracewin_env.tracewin.visualization."
                    "tracewin_distribution_from_dst",
                    return_value=distribution,
                ),
                mock.patch(
                    "beam_optimization.env.tracewin_env.tracewin.visualization."
                    "plot_tracewin_distribution_with_state",
                    return_value=object(),
                ) as plot_distribution,
            ):
                env.render_final_beam_distribution(axis_range_mm=15.0)

        kwargs = plot_distribution.call_args.kwargs
        self.assertEqual(kwargs["xy_range_mm"], 15.0)
        self.assertEqual(kwargs["angle_range_mrad"], 15.0)


if __name__ == "__main__":
    unittest.main()
