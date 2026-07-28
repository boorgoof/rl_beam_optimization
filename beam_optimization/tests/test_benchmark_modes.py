"""Command-line mode selection for the benchmark runner."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from beam_optimization.scripts import benchmark


class BenchmarkModeTests(unittest.TestCase):
    def test_tracewin_only_requires_tracewin_project(self):
        with patch.object(sys, "argv", ["benchmark", "--tracewin-only"]):
            with self.assertRaises(SystemExit) as raised:
                benchmark.main()
        self.assertEqual(raised.exception.code, 2)

    def test_policy_tracewin_only_skips_baselines_and_surrogate_policies(self):
        surrogate = MagicMock()
        dataset = MagicMock()
        tracewin_result = {"summary": {"sac": {"episodes": 1}}}

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark.json"
            argv = [
                "benchmark",
                "--policy-only",
                "--tracewin-only",
                "--tracewin",
                "project.ini",
                "--sac",
                "sac_agent.pt",
                "--output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(benchmark.ModularMLP, "load", return_value=surrogate),
                patch.object(benchmark.BeamDataset, "load", return_value=dataset),
                patch.object(benchmark, "run_bo") as run_bo,
                patch.object(benchmark, "run_svg") as run_svg,
                patch.object(
                    benchmark,
                    "run_policy_benchmark",
                    return_value=tracewin_result,
                ) as run_policy_benchmark,
                patch.object(benchmark, "save_summary_plot") as save_summary_plot,
            ):
                benchmark.main()

            run_bo.assert_not_called()
            run_svg.assert_not_called()
            save_summary_plot.assert_not_called()
            self.assertEqual(run_policy_benchmark.call_count, 1)
            call = run_policy_benchmark.call_args
            self.assertEqual(call.kwargs["tag"], "_tracewin")
            self.assertEqual(call.kwargs["episodes"], 5)
            self.assertTrue(call.kwargs["env_factory"])

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {"policy_evaluation_tracewin", "score_function"},
            )
            self.assertEqual(payload["policy_evaluation_tracewin"], tracewin_result)


if __name__ == "__main__":
    unittest.main()
