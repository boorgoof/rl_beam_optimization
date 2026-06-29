"""Flat beam dataset used to train and fine-tune the surrogate model.

BeamDataset is deliberately storage-only: it does not know how TraceWin is run
and it does not accept BeamSimulationResult objects directly. Callers append
already converted flat samples.

On disk and in memory the native format is:

    X:      Tensor (N, 25)
            columns 0..8  = initial beam state features
            columns 9..24 = machine parameters in adige.PARAMETERS order
    Y:      Tensor (N, 99)
            11 output stages flattened as 11 * 9 beam-state features
    scores: Tensor (N,)
            scalar score of the final beam state

ModularMLP expects stage-wise tensors. That conversion happens only in
get_training_batch(), right before training/evaluation needs it.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from beam_optimization.config.adige import (
    BEAM_STATE_FEATURES,
    STAGE_MARKERS,
    STAGE_PARAM_SIZES,
    score_from_vec,
)


_X_COLS = list(BEAM_STATE_FEATURES) + [
    "AD.SO.01", "AD.SO.02", "AD.ST.04.X", "AD.ST.04.Y",
    "AD.1EQ.01", "AD.1EQ.02", "AD.D.02",
    "AD.EM.6", "AD.EM.8", "AD.EM.10", "AD.EM.12",
    "AD.D.03", "AD.1EQ.03", "AD.1EQ.04",
    "AD.ST.05.X", "AD.ST.05.Y",
]
_Y_COLS = [f"{v}_s{s}" for s in range(1, 12) for v in BEAM_STATE_FEATURES]


class BeamDataset(TorchDataset):
    """Flat dataset for surrogate training and online fine-tuning."""

    def __init__(self):
        self._X: torch.Tensor = torch.empty((0, 25), dtype=torch.float32)
        self._Y: torch.Tensor = torch.empty((0, 99), dtype=torch.float32)
        self._scores: torch.Tensor = torch.empty(0, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._X.shape[0])

    def __getitem__(self, idx: int):
        """Return one sample as (stage_params, beam_states, score)."""
        stage_params, beam_states = self.get_training_batch([idx])
        stage_params = [tensor.squeeze(0) for tensor in stage_params]
        beam_states = [tensor.squeeze(0) for tensor in beam_states]
        return stage_params, beam_states, self._scores[idx]

    @property
    def X(self) -> torch.Tensor:
        """Flat input tensor with shape (N, 25)."""
        return self._X

    @property
    def Y(self) -> torch.Tensor:
        """Flat target tensor with shape (N, 99)."""
        return self._Y

    @property
    def scores(self) -> torch.Tensor:
        """Final-beam score tensor with shape (N,)."""
        return self._scores

    def append_flat_sample(self, x, y, score) -> None:
        """Append one already-converted flat sample."""
        self.append_flat_samples(
            np.asarray(x, dtype=np.float32).reshape(1, 25),
            np.asarray(y, dtype=np.float32).reshape(1, 99),
            np.asarray([score], dtype=np.float32),
        )

    def append_flat_samples(self, X, Y, scores) -> int:
        """Append a batch of already-converted flat samples."""
        X_t = torch.as_tensor(X, dtype=torch.float32)
        Y_t = torch.as_tensor(Y, dtype=torch.float32)
        S_t = torch.as_tensor(scores, dtype=torch.float32)

        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(0)
        if Y_t.ndim == 1:
            Y_t = Y_t.unsqueeze(0)
        if S_t.ndim == 0:
            S_t = S_t.unsqueeze(0)
        else:
            S_t = S_t.reshape(-1)

        if X_t.ndim != 2 or X_t.shape[1] != 25:
            raise ValueError(f"X must have shape (N, 25), got {tuple(X_t.shape)}")
        if Y_t.ndim != 2 or Y_t.shape[1] != 99:
            raise ValueError(f"Y must have shape (N, 99), got {tuple(Y_t.shape)}")
        if X_t.shape[0] != Y_t.shape[0] or X_t.shape[0] != S_t.shape[0]:
            raise ValueError(
                "X, Y, and scores must contain the same number of samples: "
                f"got {X_t.shape[0]}, {Y_t.shape[0]}, {S_t.shape[0]}"
            )

        self._X = torch.cat([self._X, X_t], dim=0)
        self._Y = torch.cat([self._Y, Y_t], dim=0)
        self._scores = torch.cat([self._scores, S_t], dim=0)
        return int(X_t.shape[0])

    def get_training_batch(
        self,
        indices,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return (stage_params, beam_states) in ModularMLP stage-wise format."""
        if not isinstance(indices, (list, np.ndarray, torch.Tensor)):
            indices = list(indices)

        X_b = self._X[indices]
        Y_b = self._Y[indices]

        params = X_b[:, 9:]
        stage_params = []
        offset = 0
        for size in STAGE_PARAM_SIZES:
            stage_params.append(params[:, offset:offset + size].contiguous())
            offset += size

        beam_states = [X_b[:, :9].contiguous()]
        for stage_idx in range(11):
            start = stage_idx * 9
            beam_states.append(Y_b[:, start:start + 9].contiguous())

        return stage_params, beam_states

    def get_initial_beam_states(self) -> torch.Tensor:
        """Return all initial beam states as a tensor with shape (N, 9)."""
        return self._X[:, :9]

    def get_param_vecs(self) -> torch.Tensor:
        """Return all flat parameter vectors as a tensor with shape (N, 16)."""
        return self._X[:, 9:]

    @classmethod
    def load(cls, path: str | Path) -> "BeamDataset":
        """Load a flat .pt dataset, with support for the legacy modular format."""
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        ds = cls()

        if "X" in raw and "Y" in raw:
            ds._X = raw["X"].float()
            ds._Y = raw["Y"].float()
            n_samples = ds._X.shape[0]
            if "scores" in raw:
                ds._scores = raw["scores"].float().reshape(-1)
            else:
                ds._scores = torch.tensor(
                    [score_from_vec(ds._Y[i, -9:].numpy()) for i in range(n_samples)],
                    dtype=torch.float32,
                )
        elif "parameter_stage_tensors" in raw:
            stage_tensors = raw["parameter_stage_tensors"]
            beam_tensors = raw["beam_state_stage_tensors"]
            n_samples = int(raw.get("num_samples", stage_tensors[0].shape[0]))

            beam0 = beam_tensors[0].float()
            params_flat = torch.cat([tensor.float() for tensor in stage_tensors], dim=1)
            ds._X = torch.cat([beam0, params_flat], dim=1)
            ds._Y = torch.cat(
                [beam_tensors[j].float() for j in range(1, 12)],
                dim=1,
            )

            if "scores" in raw:
                ds._scores = raw["scores"].float().reshape(-1)
            else:
                ds._scores = torch.tensor(
                    [score_from_vec(ds._Y[i, -9:].numpy()) for i in range(n_samples)],
                    dtype=torch.float32,
                )
        else:
            raise ValueError(
                f"Unknown .pt dataset format in {path}. Expected keys 'X'/'Y' "
                "or legacy key 'parameter_stage_tensors'."
            )

        print(f"[BeamDataset] {len(ds):,} samples loaded from {path}")
        return ds

    def merge(self, other: "BeamDataset") -> "BeamDataset":
        """Return a new dataset with samples from self followed by other."""
        merged = BeamDataset()
        merged._X = torch.cat([self._X, other._X], dim=0)
        merged._Y = torch.cat([self._Y, other._Y], dim=0)
        merged._scores = torch.cat([self._scores, other._scores], dim=0)
        return merged

    def save_flat(self, path: str | Path) -> None:
        """Save the dataset in the flat .pt format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "X": self._X,
                "Y": self._Y,
                "scores": self._scores,
                "x_cols": _X_COLS,
                "y_cols": _Y_COLS,
                "markers": list(STAGE_MARKERS),
                "num_samples": len(self),
            },
            str(path),
        )
        print(f"[BeamDataset] {len(self):,} samples saved to {path}")


SurrogateTrainingDataset = BeamDataset
