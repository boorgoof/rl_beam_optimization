"""Shared helpers for building flat beam datasets."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from beam_optimization.config.adige import BEAM_STATE_DIM, N_OUTPUT_STAGES, params_to_vec
from beam_optimization.env.simulation import BeamSimulationResult

_N_STAGES = N_OUTPUT_STAGES + 1  # output stages plus the input beam0 stage


def tracewin_result_to_flat_sample(
    result: BeamSimulationResult,
) -> Tuple[np.ndarray, np.ndarray, np.float32]:
    """Convert one dataset-eligible TraceWin result to BeamDataset format.

    Returns:
        x: (9 + N_PARAMS,) float32 = beam_state_0 (9) + machine parameters.
        y: (N_OUTPUT_STAGES * 9,) float32 = output beam states flattened.
        score: final score as float32.
    """
    if result.source != "tracewin":
        raise ValueError(f"Expected a TraceWin result, got source={result.source!r}")
    metadata = getattr(result, "metadata", {}) or {}
    physics_failure = bool(metadata.get("physics_failure"))
    encoded_failure = physics_failure and bool(
        metadata.get("failure_beam_encoded")
    )
    if (not result.success and not encoded_failure) or result.beam_states is None:
        raise ValueError(
            "Invalid TraceWin result: expected success=True or an encoded physics failure"
        )

    beam_states = np.asarray(result.beam_states, dtype=np.float32)
    if beam_states.shape != (_N_STAGES, BEAM_STATE_DIM):
        raise ValueError(
            f"TraceWin beam_states must have shape ({_N_STAGES}, {BEAM_STATE_DIM}), "
            f"got {beam_states.shape}"
        )
    if encoded_failure and not np.all(beam_states[-1] == 0.0):
        raise ValueError("Encoded TraceWin physics failure must have a zero final stage")

    x = np.concatenate([beam_states[0], params_to_vec(result.params)]).astype(np.float32)
    y = beam_states[1:].reshape(-1).astype(np.float32)
    return x, y, np.float32(result.score_val)
