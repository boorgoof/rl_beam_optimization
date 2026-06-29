"""
Convert TraceWin simulation results directly into the flat ML dataset format.

This module takes BeamSimulationResult objects produced by TraceWin, extracts
the fields needed by the surrogate dataset, and saves them as a flat ``.pt`` file.

The saved ``.pt`` file is a dictionary with this structure:

    {
        "X":           Tensor, shape (N, 25),
        "Y":           Tensor, shape (N, 99),
        "scores":      Tensor, shape (N,),
        "x_cols":      list[str], length 25,
        "y_cols":      list[str], length 99,
        "markers":     list[int], length 12,
        "num_samples": int,
    }

where each row is one TraceWin simulation:

    X[i] = beam_state_0(9 features) + machine params(16)
    Y[i] = beam_states for stages 1..11 flattened as 11 * 9 features
    scores[i] = final scalar score for that simulation

Example:
    from beam_optimization.env.tracewin_env.dataset.updated.collector import (
        append_sim_results,
    )

    results = [
        tracewin_simulator.simulate(params_1),
        tracewin_simulator.simulate(params_2),
    ]
    n_added = append_sim_results(
        results,
        "beam_optimization/env/tracewin_env/dataset/updated/collected.pt",
    )

``append_sim_results`` converts each successful BeamSimulationResult into one
new dataset row and returns the number of rows actually written.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from beam_optimization.config.adige import (
    PARAMETERS, BEAM_STATE_FEATURES, STAGE_MARKERS, params_to_vec,
)
from beam_optimization.env.simulation import BeamSimulationResult

# Column metadata saved inside the .pt file, matching dataset_train.pt.
X_COLS: List[str] = list(BEAM_STATE_FEATURES) + [p.name for p in PARAMETERS]
Y_COLS: List[str] = [f"{v}_s{s}" for s in range(1, 12) for v in BEAM_STATE_FEATURES]
MARKERS: List[int] = list(STAGE_MARKERS)


def sim_result_to_xy(result: BeamSimulationResult) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one BeamSimulationResult into one dataset sample.

    Args:
        result: BeamSimulationResult returned by TraceWinSimulator.simulate().
                It must have result.success=True and result.beam_states with
                shape (12, 9).

    Returns:
        x: (25,) float32 array: beam_state_0 (9) + machine params (16).
        y: (99,) float32 array: beam states from stages 1..11 flattened
           as 11 stages * 9 features.

    Raises:
        ValueError: if the simulation failed or has no beam_states.
    """
    # Check that the simulation was successful and has valid beam states.
    if not result.success or result.beam_states is None:
        raise ValueError("Invalid BeamSimulationResult: success=False or beam_states=None")

    # Convert the beam states to float32 
    beam_states = result.beam_states.astype(np.float32)  # (12, 9)

    # Create the input vector x by concatenating the initial beam state and the machine parameters.
    x = np.concatenate([
        beam_states[0],                # initial beam state: stage 0, shape (9,)
        params_to_vec(result.params),  # ordered machine parameters, shape (16,)
    ]).astype(np.float32)              # (25,)

    # Create the output vector y by flattening the beam states from stages 1..11.
    y = beam_states[1:].flatten().astype(np.float32)  # stages 1..11 -> (99,)

    return x, y


def append_sim_results(
    results: List[BeamSimulationResult],
    path: str | Path,
    *,
    skip_failed: bool = True,
) -> int:
    """Append BeamSimulationResult samples to a flat .pt dataset file.

    Args:
        results:     BeamSimulationResult objects produced by TraceWin.
        path:        Destination .pt file. It is created if it does not exist.
        skip_failed: If True, ignore failed simulations.

    Returns:
        Number of samples actually appended.
    """
    path = Path(path)

    # Convert valid simulation results into new rows.
    new_x, new_y, new_scores = [], [], []
    for r in results:
        if skip_failed and (not r.success or r.beam_states is None):
            continue
        try:
            x, y = sim_result_to_xy(r)
        except ValueError:
            continue
        new_x.append(x)
        new_y.append(y)
        new_scores.append(np.float32(r.score_val))

    if not new_x:
        return 0

    # Stack the new rows into tensors.
    new_X = torch.tensor(np.stack(new_x), dtype=torch.float32)   # (k, 25)
    new_Y = torch.tensor(np.stack(new_y), dtype=torch.float32)   # (k, 99)
    new_S = torch.tensor(new_scores, dtype=torch.float32)         # (k,)

    # Load the existing dataset if it exists, and append the new rows.
    if path.exists():
        # Load the existing dataset and append the new rows.
        existing = torch.load(str(path), map_location="cpu", weights_only=False)
        X = torch.cat([existing["X"].float(), new_X], dim=0)
        Y = torch.cat([existing["Y"].float(), new_Y], dim=0)
        S = torch.cat([existing.get("scores", torch.empty(0)).float(), new_S], dim=0)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        X, Y, S = new_X, new_Y, new_S

    # Save the updated dataset.
    torch.save({
        "X":           X,
        "Y":           Y,
        "scores":      S,
        "x_cols":      X_COLS,
        "y_cols":      Y_COLS,
        "markers":     MARKERS,
        "num_samples": X.shape[0],
    }, str(path))

    n_added = len(new_x)
    print(f"[collector] +{n_added} samples -> {path}  (total: {X.shape[0]})")
    return n_added


def create_flat_dataset(
    results: List[BeamSimulationResult],
    path: str | Path,
    *,
    skip_failed: bool = True,
) -> int:
    """Create a new flat .pt dataset, overwriting the existing file if needed.

    This is equivalent to deleting the file first and then calling
    append_sim_results().
    """
    path = Path(path)
    if path.exists():
        path.unlink()
    return append_sim_results(results, path, skip_failed=skip_failed)
