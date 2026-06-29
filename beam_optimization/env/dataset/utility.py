"""Shared helpers for building flat beam datasets."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from beam_optimization.config.adige import params_to_vec
from beam_optimization.env.simulation import BeamSimulationResult


def tracewin_result_to_flat_sample(
    result: BeamSimulationResult,
) -> Tuple[np.ndarray, np.ndarray, np.float32]:
    """Convert one successful TraceWin result to BeamDataset X/Y/score format.

    Returns:
        x: (25,) float32 = beam_state_0 (9) + machine parameters (16).
        y: (99,) float32 = output beam states 1..11 flattened.
        score: final score as float32.
    """
    if result.source != "tracewin":
        raise ValueError(f"Expected a TraceWin result, got source={result.source!r}")
    if not result.success or result.beam_states is None:
        raise ValueError("Invalid TraceWin result: success=False or beam_states=None")

    beam_states = np.asarray(result.beam_states, dtype=np.float32)
    if beam_states.shape != (12, 9):
        raise ValueError(
            "TraceWin beam_states must have shape (12, 9), "
            f"got {beam_states.shape}"
        )

    x = np.concatenate([beam_states[0], params_to_vec(result.params)]).astype(np.float32)
    y = beam_states[1:].reshape(-1).astype(np.float32)
    return x, y, np.float32(result.score_val)
