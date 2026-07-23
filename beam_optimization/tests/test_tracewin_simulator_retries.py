from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    ERROR_SCORE,
    N_STAGES,
    PARAMETERS,
)
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
        self.assertTrue(result.metadata["physics_failure"])
        self.assertFalse(result.metadata["failure_beam_encoded"])
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

    def test_physics_failure_uses_cached_real_beam0(self):
        with tempfile.TemporaryDirectory() as directory:
            simulator = self._simulator(directory)
            beam0 = np.arange(1, BEAM_STATE_DIM + 1, dtype=np.float32)
            simulator._cached_input_beam = beam0.copy()

            result = simulator._physics_failure_result(
                {PARAMETERS[0].key: 0.123},
                "Error: All particles are lost",
            )

        self.assertFalse(result.success)
        self.assertEqual(result.score_val, ERROR_SCORE)
        self.assertEqual(result.beam_states.shape, (N_STAGES, BEAM_STATE_DIM))
        np.testing.assert_array_equal(result.beam_states[0], beam0)
        np.testing.assert_array_equal(result.beam_states[1:], 0.0)
        self.assertEqual(
            result.final_beam,
            {feature: 0.0 for feature in BEAM_STATE_FEATURES},
        )
        self.assertTrue(result.metadata["failure_beam_encoded"])
        self.assertEqual(result.metadata["beam0_source"], "cached")

    def test_physics_failure_prefers_part_rfq_over_cached_beam0(self):
        with tempfile.TemporaryDirectory() as directory:
            simulator = self._simulator(directory)
            input_dst = Path(simulator.calc_dir) / "part_rfq.dst"
            input_dst.write_bytes(b"synthetic")
            from_dst = np.linspace(0.1, 0.9, BEAM_STATE_DIM, dtype=np.float32)
            simulator._cached_input_beam = np.full(BEAM_STATE_DIM, 99.0, dtype=np.float32)

            with mock.patch(
                "beam_optimization.env.tracewin_env.tracewin.tracewin_simulator._beam0_from_dst",
                return_value=from_dst,
            ) as reader:
                result = simulator._physics_failure_result(
                    {PARAMETERS[0].key: 0.123},
                    "Error: All particles are lost",
                )

        reader.assert_called_once_with(input_dst, simulator.initial_npart)
        np.testing.assert_array_equal(result.beam_states[0], from_dst)
        np.testing.assert_array_equal(simulator._cached_input_beam, from_dst)
        self.assertEqual(result.metadata["beam0_source"], "part_rfq.dst")

    def test_physics_failure_preserves_available_partran_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            simulator = self._simulator(directory)
            rows = []
            for marker, npart, fill in (
                (0, 10_000, 1.0),
                (2, 9_000, 2.0),
                (195, 5_000, 3.0),
                (197, 500, 0.0),
            ):
                row = {"##": marker, "npart": npart}
                row.update({
                    feature: fill
                    for feature in BEAM_STATE_FEATURES
                    if feature != "npart_ratio"
                })
                rows.append(row)
            partial_df = pd.DataFrame(rows)
            tracewin = mock.Mock()
            tracewin.results.return_value = partial_df

            result = simulator._physics_failure_result(
                {PARAMETERS[0].key: 0.123},
                "Error: All particles are lost",
                tracewin,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.score_val, ERROR_SCORE)
        self.assertEqual(result.metadata["beam0_source"], "partran1.out")
        self.assertEqual(result.metadata["available_stage_markers"], [0, 2, 195, 197])
        self.assertEqual(result.metadata["last_available_marker"], 197)
        self.assertEqual(result.metadata["n_available_output_stages"], 3)
        self.assertTrue(np.any(result.beam_states[1] != 0.0))
        self.assertTrue(np.any(result.beam_states[6] != 0.0))
        self.assertEqual(result.beam_states[7, 0], 0.05)
        np.testing.assert_array_equal(result.beam_states[8:], 0.0)


if __name__ == "__main__":
    unittest.main()
