"""Color semantics used by BaseBeamEnv.render()."""
from __future__ import annotations

import unittest

from beam_optimization.env.base_beam_env import BaseBeamEnv


class RenderColorTests(unittest.TestCase):
    def test_knn_distance_trend_is_always_blue(self):
        colors = BaseBeamEnv._segment_colors(
            [1.0, 2.0, 0.5, 0.5],
            feature="knn_distance",
        )

        self.assertEqual(colors, ["tab:blue", "tab:blue", "tab:blue"])

    def test_score_trend_keeps_improvement_colors(self):
        colors = BaseBeamEnv._segment_colors([1.0, 2.0, 0.5], feature=None)

        self.assertEqual(colors, ["tab:green", "tab:red"])


if __name__ == "__main__":
    unittest.main()
