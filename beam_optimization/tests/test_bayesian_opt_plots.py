from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from beam_optimization.config.adige import PARAMETERS
from beam_optimization.scripts.bayesian_opt import (
    _delta_plot_score_summary,
    _format_delta_plot_label,
    save_convergence_plot,
    save_delta_plot,
)


class BayesianPlotTests(unittest.TestCase):
    def _best_result(self) -> dict:
        params = {}
        for index, parameter in enumerate(PARAMETERS):
            direction = 1.0 if index % 2 == 0 else -1.0
            params[parameter.key] = (
                float(parameter.default)
                + direction * 0.25 * float(parameter.sensitivity)
            )
        return {
            "origin": "online_tracewin",
            "score": 12.2578,
            "params": params,
            "success": True,
        }

    def test_delta_labels_include_normalized_and_physical_values(self):
        self.assertEqual(
            _format_delta_plot_label(0.25, 0.00125),
            "+0.25 sens\nΔ=+0.00125",
        )

    def test_score_summary_includes_best_default_and_improvement(self):
        summary = _delta_plot_score_summary(
            self._best_result(),
            {"score": 10.0, "success": True},
        )
        self.assertEqual(
            summary,
            "Best score = 12.2578 | Default score = 10 | Δscore = +2.2578",
        )

    def test_bayesian_png_plots_are_created(self):
        report = {
            "runs": [
                {
                    "evaluations": [
                        {"score": -2.0},
                        {"score": 1.5},
                        {"score": 1.0},
                        {"score": 4.25},
                    ]
                },
                {
                    "evaluations": [
                        {"score": 0.5},
                        {"score": 3.75},
                    ]
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            delta_path = save_delta_plot(
                self._best_result(),
                output,
                default_result={"score": 10.0, "success": True},
            )
            convergence_path = save_convergence_plot(report, output)

            self.assertTrue(delta_path.is_file())
            self.assertGreater(delta_path.stat().st_size, 1000)
            self.assertIsNotNone(convergence_path)
            self.assertTrue(convergence_path.is_file())
            self.assertGreater(convergence_path.stat().st_size, 1000)

    def test_notebook_compares_existing_scores_without_another_simulation(self):
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "env/tracewin_env/tracewin/visualize_distributions_python_run.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        comparison_cells = [
            cell
            for cell in notebook["cells"]
            if cell.get("id") == "score-comparison-plot"
        ]
        self.assertEqual(len(comparison_cells), 1)
        source = "".join(comparison_cells[0]["source"])
        self.assertIn("python_result.score_val", source)
        self.assertIn("default_result.score_val", source)
        self.assertIn("python_hardcoded_vs_default_score.png", source)
        self.assertNotIn(".simulate(", source)


if __name__ == "__main__":
    unittest.main()
