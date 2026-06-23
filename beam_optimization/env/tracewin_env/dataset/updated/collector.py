"""
collector.py — converte risultati TraceWin direttamente in formato flat ml_dataset.

Questo è il pezzo che prende i BeamSimulationResult prodotti da TraceWin, estrae le
feature necessarie (beam_state_0 + 16 params come X, beam_states 1..11 come Y)
e li salva come flat .pt senza passare per il formato modular intermedio.

Utilizzo tipico:
    from beam_optimization.env.tracewin_env.ml_dataset.collector import (
        sim_result_to_xy, append_sim_results
    )

    x, y = sim_result_to_xy(result)           # (25,), (99,)
    n = append_sim_results([r1, r2], path)    # aggiunge a ml_dataset/collected.pt
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from beam_optimization.config.adige import (
    PARAMETERS, PARAM_KEYS, BEAM_STATE_VARS, STAGE_MARKERS, params_to_vec,
)
from beam_optimization.env.simulation import BeamSimulationResult

# Metadati colonne (stesso formato di dataset_train.pt)
X_COLS: List[str] = list(BEAM_STATE_VARS) + [p.name for p in PARAMETERS]
Y_COLS: List[str] = [f"{v}_s{s}" for s in range(1, 12) for v in BEAM_STATE_VARS]
MARKERS: List[int] = list(STAGE_MARKERS)


def sim_result_to_xy(result: BeamSimulationResult) -> Tuple[np.ndarray, np.ndarray]:
    """Converte un BeamSimulationResult in una coppia (x, y) per il dataset flat.

    Args:
        result: BeamSimulationResult da TraceWinSimulator.simulate().
                Deve avere result.success=True e result.beam_states.shape=(12,9).

    Returns:
        x: (25,) float32 — beam_state_0 (9) + parametri flat (16)
        y: (99,) float32 — beam_states agli stage 1..11 concatenati (11×9)

    Raises:
        ValueError: se result.success è False o beam_states è None.
    """
    if not result.success or result.beam_states is None:
        raise ValueError("BeamSimulationResult non valido: success=False o beam_states=None")

    bs = result.beam_states.astype(np.float32)  # (12, 9)

    x = np.concatenate([
        bs[0],                    # beam_state_0 (9,)
        params_to_vec(result.params),  # parametri ordinati (16,)
    ]).astype(np.float32)         # (25,)

    y = bs[1:].flatten().astype(np.float32)  # stages 1..11 → (99,)

    return x, y


def append_sim_results(
    results: List[BeamSimulationResult],
    path: str | Path,
    *,
    skip_failed: bool = True,
) -> int:
    """Aggiunge BeamSimulationResult a un flat .pt esistente (o lo crea se non esiste).

    Args:
        results:      Lista di BeamSimulationResult da TraceWin.
        path:         Path del file .pt di destinazione.
        skip_failed:  Se True (default), ignora i risultati con success=False.

    Returns:
        Numero di campioni effettivamente aggiunti.
    """
    path = Path(path)

    # Raccoglie le nuove righe
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

    new_X = torch.tensor(np.stack(new_x), dtype=torch.float32)   # (k, 25)
    new_Y = torch.tensor(np.stack(new_y), dtype=torch.float32)   # (k, 99)
    new_S = torch.tensor(new_scores, dtype=torch.float32)         # (k,)

    if path.exists():
        # Carica esistente e concatena
        existing = torch.load(str(path), map_location="cpu", weights_only=False)
        X = torch.cat([existing["X"].float(), new_X], dim=0)
        Y = torch.cat([existing["Y"].float(), new_Y], dim=0)
        S = torch.cat([existing.get("scores", torch.empty(0)).float(), new_S], dim=0)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        X, Y, S = new_X, new_Y, new_S

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
    print(f"[collector] +{n_added} campioni → {path}  (totale: {X.shape[0]})")
    return n_added


def create_flat_dataset(
    results: List[BeamSimulationResult],
    path: str | Path,
    *,
    skip_failed: bool = True,
) -> int:
    """Crea un nuovo flat .pt sovrascrivendo quello esistente.

    Equivalente a cancellare il file e chiamare append_sim_results.
    """
    path = Path(path)
    if path.exists():
        path.unlink()
    return append_sim_results(results, path, skip_failed=skip_failed)
