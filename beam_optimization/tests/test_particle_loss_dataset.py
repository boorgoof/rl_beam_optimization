from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    ERROR_SCORE,
    N_STAGES,
    default_params,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.dataset.tracewin_dataset_builder import (
    TraceWinDatasetBuilder,
    _is_valid_tracewin_result,
)
from beam_optimization.env.dataset.utility import tracewin_result_to_flat_sample
from beam_optimization.env.simulation import BeamSimulationResult


def _result(*, success: bool, encoded: bool = False, technical: bool = False):
    params = default_params()
    if technical:
        return BeamSimulationResult(
            params=params,
            beam_states=None,
            final_beam=None,
            score_val=ERROR_SCORE,
            success=False,
            source="tracewin",
            error="Qt platform plugin failed",
        )

    beam_states = np.ones((N_STAGES, BEAM_STATE_DIM), dtype=np.float32)
    beam_states[0] = np.linspace(0.1, 0.9, BEAM_STATE_DIM, dtype=np.float32)
    if encoded:
        beam_states[1:] = 0.0
        beam_states[1:4] = np.arange(
            3 * BEAM_STATE_DIM,
            dtype=np.float32,
        ).reshape(3, BEAM_STATE_DIM) + 1.0
    return BeamSimulationResult(
        params=params,
        beam_states=beam_states,
        final_beam=None,
        score_val=ERROR_SCORE if encoded else 12.0,
        success=success,
        source="tracewin",
        error="Error: All particles are lost" if encoded else None,
        metadata={
            "physics_failure": encoded,
            "failure_beam_encoded": encoded,
        },
    )


class _SequenceSimulator:
    def __init__(self, results):
        self.results = list(results)

    def simulate(self, params):
        result = self.results.pop(0)
        result.params = dict(params)
        return result


class ParticleLossDatasetTests(unittest.TestCase):
    def test_flat_sample_preserves_beam0_and_available_partial_outputs(self):
        result = _result(success=False, encoded=True)
        x, y, stored_score = tracewin_result_to_flat_sample(result)

        np.testing.assert_array_equal(x[:BEAM_STATE_DIM], result.beam_states[0])
        np.testing.assert_array_equal(
            y,
            result.beam_states[1:].reshape(-1),
        )
        self.assertTrue(np.any(y != 0.0))
        np.testing.assert_array_equal(y[-BEAM_STATE_DIM:], 0.0)
        self.assertEqual(stored_score, ERROR_SCORE)
        self.assertTrue(_is_valid_tracewin_result(result))

    def test_technical_failure_is_not_dataset_eligible(self):
        result = _result(success=False, technical=True)
        self.assertFalse(_is_valid_tracewin_result(result))
        with self.assertRaisesRegex(ValueError, "encoded physics failure"):
            tracewin_result_to_flat_sample(result)

    def test_builder_accepts_physics_failure_and_reports_outcomes(self):
        simulator = _SequenceSimulator(
            [
                _result(success=False, encoded=True),
                _result(success=False, technical=True),
                _result(success=True),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "dataset"
            report = TraceWinDatasetBuilder(
                simulator,
                output_dir=output_dir,
                target_samples=2,
                seed=123,
            ).build()
            dataset = BeamDataset.load(output_dir / "dataset_all.pt")

        self.assertEqual(len(dataset), 2)
        self.assertEqual(report["n_accepted"], 2)
        self.assertEqual(report["n_success"], 1)
        self.assertEqual(report["n_physics_failures"], 1)
        self.assertEqual(report["n_technical_failures"], 1)
        self.assertEqual(report["n_failed"], 2)
        self.assertTrue(np.any(dataset.Y[0].numpy() != 0.0))
        np.testing.assert_array_equal(
            dataset.Y[0, -BEAM_STATE_DIM:].numpy(),
            0.0,
        )
        self.assertEqual(dataset.scores[0].item(), ERROR_SCORE)


if __name__ == "__main__":
    unittest.main()
