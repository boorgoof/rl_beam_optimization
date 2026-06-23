"""
SurrogateTrainingDataset — dati di simulazione per allenare il surrogate.

Storage interno: flat tensors X(N,25) e Y(N,99), identici al formato
ml_dataset su disco.  La conversione stage-wise per ModularMLP avviene
solo in get_training_batch(), esclusivamente quando il surrogate ne ha bisogno.

Formato X (N, 25):
    [:, 0:9]  = beam_state_0  (9 variabili BEAM_STATE_VARS)
    [:, 9:25] = parametri flat (16, ordine di PARAMETERS in adige.py)

Formato Y (N, 99):
    [:, s*9:(s+1)*9] = beam_state allo stage s+1  (s = 0..10)

Caricamento supportato:
    - Flat (.pt con chiavi "X","Y") — formato ml_dataset, nativo
    - Modular (.pt con chiave "parameter_stage_tensors") — formato legacy,
      convertito automaticamente al caricamento (deprecato)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from beam_optimization.config.adige import (
    BEAM_STATE_VARS, BEAM_STATE_DIM, N_STAGES, N_BEAM_STATE_STAGES,
    STAGE_PARAM_SIZES, params_to_vec, score_from_vec,
)
from beam_optimization.env.simulation import BeamSimulationResult

# Nomi colonne del formato flat (usati come metadati nel .pt)
_X_COLS = list(BEAM_STATE_VARS) + [
    "AD.SO.01", "AD.SO.02", "AD.ST.04.X", "AD.ST.04.Y",
    "AD.1EQ.01", "AD.1EQ.02", "AD.D.02",
    "AD.EM.6", "AD.EM.8", "AD.EM.10", "AD.EM.12",
    "AD.D.03", "AD.1EQ.03", "AD.1EQ.04",
    "AD.ST.05.X", "AD.ST.05.Y",
]
_Y_COLS = [f"{v}_s{s}" for s in range(1, 12) for v in BEAM_STATE_VARS]
from beam_optimization.config.adige import STAGE_MARKERS as _MARKERS


class SurrogateTrainingDataset(TorchDataset):
    """Dataset per training e fine-tuning del surrogate.

    Archivia i dati nativamente come tensori flat X(N,25) e Y(N,99),
    identici al formato ml_dataset su disco.  Espone get_training_batch()
    per la conversione stage-wise usata dal surrogate.
    """

    def __init__(self):
        # Storage principale flat
        self._X: torch.Tensor = torch.empty((0, 25), dtype=torch.float32)
        self._Y: torch.Tensor = torch.empty((0, 99), dtype=torch.float32)
        self._scores: torch.Tensor = torch.empty(0, dtype=torch.float32)

    # ── Lunghezza ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._X.shape[0]

    # ── TorchDataset interface (compatibilità DataLoader) ──────────────────────

    def __getitem__(self, idx: int):
        """Restituisce (stage_params, beam_states, score) per un campione."""
        stage_params, beam_states = self.get_training_batch([idx])
        stage_params = [t.squeeze(0) for t in stage_params]
        beam_states  = [t.squeeze(0) for t in beam_states]
        sc = self._scores[idx]
        return stage_params, beam_states, sc

    # ── Aggiunta dati (fine-tuning online) ────────────────────────────────────

    def add(self, result: BeamSimulationResult) -> bool:
        """Aggiunge un risultato di simulazione valido al dataset."""
        if not result.success or result.beam_states is None:
            return False

        # Costruisce X e Y direttamente dal risultato — nessun formato intermedio
        x = np.concatenate([
            result.beam_states[0].astype(np.float32),   # beam_state_0 (9,)
            params_to_vec(result.params),                # params flat  (16,)
        ])  # (25,)
        y = result.beam_states[1:].astype(np.float32).flatten()  # (99,)

        x_t  = torch.tensor(x, dtype=torch.float32).unsqueeze(0)    # (1,25)
        y_t  = torch.tensor(y, dtype=torch.float32).unsqueeze(0)    # (1,99)
        sc_t = torch.tensor([result.score_val], dtype=torch.float32) # (1,)

        self._X      = torch.cat([self._X,      x_t],  dim=0)
        self._Y      = torch.cat([self._Y,      y_t],  dim=0)
        self._scores = torch.cat([self._scores, sc_t], dim=0)
        return True

    # ── Accesso stage-wise (per surrogate training / fine-tuning) ──────────────

    def get_training_batch(
        self, indices
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Ritorna (stage_params, beam_states) in formato stage-wise per ModularMLP.

        La conversione flat→stage avviene qui, solo quando il surrogate lo richiede.

        Args:
            indices: lista/array di indici da campionare

        Returns:
            stage_params: list di 11 tensori, shape (batch, stage_size)
            beam_states:  list di 12 tensori, shape (batch, 9)
                          [0] = beam_state_0, [1..11] = stage 1..11
        """
        if not isinstance(indices, (list, np.ndarray)):
            indices = list(indices)

        X_b = self._X[indices]   # (batch, 25)
        Y_b = self._Y[indices]   # (batch, 99)

        # Divide i 16 params in 11 gruppi per stage
        params = X_b[:, 9:]      # (batch, 16)
        stage_params = []
        offset = 0
        for sz in STAGE_PARAM_SIZES:
            stage_params.append(params[:, offset:offset + sz].contiguous())
            offset += sz

        # beam_state_0 da X, poi i 11 stage da Y
        beam_states = [X_b[:, :9].contiguous()]
        for s in range(11):
            beam_states.append(Y_b[:, s * 9:(s + 1) * 9].contiguous())

        return stage_params, beam_states  # 11 tensori, 12 tensori

    # ── Accesso agli stati iniziali (per SurrogateEnv reset) ───────────────────

    def get_initial_beam_states(self) -> torch.Tensor:
        """Restituisce tutti gli stati al stage 0 come tensore (N, 9)."""
        return self._X[:, :9]

    def get_param_vecs(self) -> torch.Tensor:
        """Restituisce tutti i vettori di parametri come tensore (N, 16)."""
        return self._X[:, 9:]

    # ── Caricamento da file ────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateTrainingDataset":
        """Carica da .pt.  Supporta il formato flat (ml_dataset) e legacy modular.

        Il formato flat è riconosciuto dalla presenza delle chiavi "X" e "Y".
        Il formato legacy (parameter_stage_tensors) viene convertito al volo.
        """
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        ds = cls()

        if "X" in raw and "Y" in raw:
            # ── Formato flat (ml_dataset) — nativo ───────────────────────────
            ds._X = raw["X"].float()
            ds._Y = raw["Y"].float()
            N = ds._X.shape[0]
            if "scores" in raw:
                ds._scores = raw["scores"].float()
            else:
                ds._scores = torch.tensor(
                    [score_from_vec(ds._Y[i, -9:].numpy()) for i in range(N)],
                    dtype=torch.float32,
                )

        elif "parameter_stage_tensors" in raw:
            # ── Formato modular legacy (tracewin_data) — conversione automatica
            stage_tensors = raw["parameter_stage_tensors"]   # list[11] × (N, size_i)
            beam_tensors  = raw["beam_state_stage_tensors"]  # list[12] × (N, 9)
            N = int(raw.get("num_samples", stage_tensors[0].shape[0]))

            beam0      = beam_tensors[0].float()                           # (N,9)
            params_flat = torch.cat([t.float() for t in stage_tensors], dim=1)  # (N,16)
            ds._X = torch.cat([beam0, params_flat], dim=1)                 # (N,25)
            ds._Y = torch.cat([beam_tensors[j].float() for j in range(1, 12)], dim=1)  # (N,99)

            if "scores" in raw:
                ds._scores = raw["scores"].float()
            else:
                ds._scores = torch.tensor(
                    [score_from_vec(ds._Y[i, -9:].numpy()) for i in range(N)],
                    dtype=torch.float32,
                )
        else:
            raise ValueError(
                f"Formato .pt non riconosciuto in {path}. "
                "Atteso: chiavi 'X','Y' (flat) oppure 'parameter_stage_tensors' (modular)."
            )

        print(f"[SurrogateTrainingDataset] {len(ds):,} campioni caricati da {path}")
        return ds

    # ── Merge ─────────────────────────────────────────────────────────────────

    def merge(self, other: "SurrogateTrainingDataset") -> "SurrogateTrainingDataset":
        """Ritorna un nuovo dataset con i campioni di self + other concatenati."""
        merged = SurrogateTrainingDataset()
        merged._X = torch.cat([self._X, other._X], dim=0)
        merged._Y = torch.cat([self._Y, other._Y], dim=0)
        merged._scores = torch.cat([self._scores, other._scores], dim=0)
        return merged

    # ── Salvataggio in formato flat ────────────────────────────────────────────

    def save_flat(self, path: str | Path) -> None:
        """Salva il dataset nel formato flat ml_dataset (.pt)."""
        torch.save({
            "X":          self._X,
            "Y":          self._Y,
            "scores":     self._scores,
            "x_cols":     _X_COLS,
            "y_cols":     _Y_COLS,
            "markers":    list(_MARKERS),
            "num_samples": len(self),
        }, str(path))
        print(f"[SurrogateTrainingDataset] {len(self):,} campioni salvati in {path}")


# Legacy alias kept so older imports and scripts continue to work.
BeamDataset = SurrogateTrainingDataset
