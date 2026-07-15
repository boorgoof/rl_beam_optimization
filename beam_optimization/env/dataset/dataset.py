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
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset as TorchDataset

from beam_optimization.config.adige import (
    BEAM_STATE_FEATURES,
    STAGE_MARKERS,
    STAGE_PARAM_SIZES,
    score_from_vec,
)
from beam_optimization.config.paths import default_dataset_path


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
        self._param_knn_tree: Optional[cKDTree] = None
        self._param_knn_std: Optional[np.ndarray] = None

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

    def param_knn_distance(self, param_vecs, k: int = 5) -> np.ndarray:
        """Mean distance from each row of `param_vecs` (N, 16) to its k nearest
        parameter vectors in this dataset.

        Distances are computed after standardizing every parameter by this
        dataset's per-parameter std, so no single parameter's scale (e.g.
        volts vs. millimeters) dominates the result. The k-d tree and std are
        built once and cached on the instance.
        """
        if self._param_knn_tree is None:
            ref = self.get_param_vecs().numpy()
            std = ref.std(axis=0)
            std[std == 0] = 1.0
            self._param_knn_std = std
            self._param_knn_tree = cKDTree(ref / std)

        query = np.atleast_2d(np.asarray(param_vecs, dtype=np.float64)) / self._param_knn_std
        dists, _ = self._param_knn_tree.query(query, k=k)
        if dists.ndim == 1:
            dists = dists[:, None]
        return dists.mean(axis=1)

    @classmethod
    def load(cls, path: str | Path) -> "BeamDataset":
        """Load a flat .pt dataset and recompute scores with current config.

        Stored scores are derived data and may have been produced by an older
        score function. Recomputing them from the final nine columns of ``Y``
        keeps existing datasets consistent whenever score shaping changes.
        """
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        if "X" not in raw or "Y" not in raw:
            raise ValueError(f"Unknown .pt dataset format in {path}. Expected keys 'X'/'Y'.")

        ds = cls()
        ds._X = raw["X"].float()
        ds._Y = raw["Y"].float()
        ds._scores = torch.tensor(
            [score_from_vec(ds._Y[i, -9:].numpy()) for i in range(len(ds._X))],
            dtype=torch.float32,
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


_default_dataset_cache: Optional[BeamDataset] = None


def param_knn_distance(param_vecs, dataset: Optional[BeamDataset] = None, k: int = 5) -> np.ndarray:
    """Mean distance from each row of `param_vecs` (N, 16) to its k nearest
    parameter vectors in `dataset` (the default dataset when not given).

    Falls back to a process-wide cached load of the default dataset when
    `dataset` is None, so repeated calls (e.g. once per env step during a
    test episode) don't reload the file from disk every time.
    """
    global _default_dataset_cache
    if dataset is None:
        if _default_dataset_cache is None:
            _default_dataset_cache = BeamDataset.load(default_dataset_path())
        dataset = _default_dataset_cache
    return dataset.param_knn_distance(param_vecs, k=k)
