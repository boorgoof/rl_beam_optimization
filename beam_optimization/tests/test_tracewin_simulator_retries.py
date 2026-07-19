from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from beam_optimization.config.adige import ERROR_SCORE, PARAMETERS
from beam_optimization.env.simulation import BeamSimulationResult
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import (
    TraceWinSimulator,
    _non_retryable_physics_failure,
)


class _FailingTraceWin:
    calls = 0
    stdout = ""
    stderr = ""

    def __init__(self, project, outpath):
        self.last_stdout = self.stdout
        self.last_stderr = self.stderr

    def run(self, **kwargs):
        type(self).calls += 1
        return False

    def results(self):
        raise AssertionError("results must not be read after a failed TraceWin run")


class TraceWinRetryTests(unittest.TestCase):
    def _simulator(self, directory: str, *, retries: int = 2) -> TraceWinSimulator:
        project = Path(directory) / "project.ini"
        project.write_text("synthetic project", encoding="utf-8")
        return TraceWinSimulator(
            project_file=str(project),
            calc_dir=str(Path(directory) / "calc"),
            retries=retries,
            retry_sleep=0.0,
            kill_stale=False,
            num_threads=1,
        )

    def _run_failure(self, stdout: str, stderr: str = ""):
        with tempfile.TemporaryDirectory() as directory:
            simulator = self._simulator(directory)
            _FailingTraceWin.calls = 0
            _FailingTraceWin.stdout = stdout
            _FailingTraceWin.stderr = stderr
            output = io.StringIO()
            with mock.patch(
                "beam_optimization.env.tracewin_env.tracewin.tracewin_simulator.TraceWin",
                _FailingTraceWin,
            ), contextlib.redirect_stdout(output):
                result = simulator.simulate({PARAMETERS[0].key: 0.123})
        return result, _FailingTraceWin.calls, output.getvalue()

    def test_all_particles_lost_returns_failed_dataclass_without_retry(self):
        original = "Error: All particles are lost"
        result, calls, output = self._run_failure(original)

        self.assertIsInstance(result, BeamSimulationResult)
        self.assertFalse(result.success)
        self.assertEqual(result.score_val, ERROR_SCORE)
        self.assertIsNone(result.beam_states)
        self.assertIsNone(result.final_beam)
        self.assertEqual(result.params[PARAMETERS[0].key], 0.123)
        self.assertEqual(result.error, original)
        self.assertEqual(calls, 1)
        self.assertNotIn("retrying", output)

    def test_synchronous_particle_failure_in_stderr_is_not_retried(self):
        original = "ERROR: SYNCHRONOUS PARTICLE NEVER REACHES THE END OF THE FIELD MAP!"
        result, calls, output = self._run_failure("", original)

        self.assertFalse(result.success)
        self.assertEqual(result.error, original)
        self.assertEqual(calls, 1)
        self.assertNotIn("retrying", output)

    def test_partial_beam_field_map_failure_is_not_retried(self):
        original = (
            "Error: Part of the beam distribution never reaches the end "
            "of the field map!"
        )
        result, calls, output = self._run_failure(original)

        self.assertFalse(result.success)
        self.assertEqual(result.error, original)
        self.assertEqual(calls, 1)
        self.assertNotIn("retrying", output)

    def test_physics_failure_matching_is_case_insensitive(self):
        self.assertEqual(
            _non_retryable_physics_failure("eRrOr: ALL PARTICLES ARE LOST", ""),
            "eRrOr: ALL PARTICLES ARE LOST",
        )
        self.assertIsNone(
            _non_retryable_physics_failure("Error: temporary SSH failure", "")
        )

    def test_technical_failure_still_uses_all_retries(self):
        result, calls, output = self._run_failure("Qt platform plugin xcb failed")

        self.assertFalse(result.success)
        self.assertEqual(calls, 3)
        self.assertEqual(output.count("retrying"), 2)
        self.assertIn("Qt platform plugin xcb failed", result.error)


if __name__ == "__main__":
    unittest.main()
