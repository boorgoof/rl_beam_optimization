"""FailureClassifier — binary classifier predicting whether a full ADIGE
parameter set (plus an initial beam state) will produce all-particles-lost
at the final stage.

ModularMLP is a continuous regressor and can never predict npart_ratio
exactly 0, so it cannot reproduce the physical cliff that score() checks via
is_all_particles_lost(). This is a separate, independently trained model --
same (stage_params, beam_state_0) input contract as ModularMLP, no shared
weights or architecture -- used to gate score()/score_tensor() calls made
directly on surrogate predictions, where the regression's inability to hit
the exact-zero cliff would otherwise produce an arbitrary score near the
boundary instead of ERROR_SCORE.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from beam_optimization.config.adige import (
    ALL_PARTICLES_LOST_NPART_RATIO,
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    STAGE_PARAM_SIZES,
)


class FailureClassifier(nn.Module):
    """Binary classifier: P(final npart_ratio <= ALL_PARTICLES_LOST_NPART_RATIO).

    Input: same (stage_params, beam_state_0) contract as ModularMLP.forward(),
    but consumed all at once (concatenated into one flat vector) rather than
    stage-by-stage -- the label depends only on the full parameter set and
    initial beam state, not on an intermediate latent trajectory.
    """

    def __init__(
        self,
        hidden_sizes: List[int] = (128, 128),
        dropout: float = 0.1,
        norm_stats: Optional[dict] = None,
    ):
        super().__init__()
        self.hidden_sizes = list(hidden_sizes)
        self.dropout = float(dropout)
        self._norm_stats = norm_stats

        input_dim = BEAM_STATE_DIM + sum(STAGE_PARAM_SIZES)
        layers: List[nn.Module] = []
        prev = input_dim
        for h in self.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        if norm_stats is not None:
            self._register_norm_buffers(norm_stats)

    def _register_norm_buffers(self, stats: dict) -> None:
        """Register mean/std buffers for beam0 (stage 0) and every stage's
        parameters, reusing the same normalization_metadata already computed
        for the regressor ensemble (compute_normalization_metadata() in
        trainer.py) -- no separate normalization pass needed."""
        def _val(x):
            return x["values"] if isinstance(x, dict) else x

        for i, (m, v) in enumerate(zip(stats["parameter_means"], stats["parameter_variances"])):
            self.register_buffer(f"pm_{i}", _val(m).float())
            self.register_buffer(f"ps_{i}", torch.sqrt(_val(v).float() + 1e-8))
        m0, v0 = stats["beam_state_means"][0], stats["beam_state_variances"][0]
        self.register_buffer("bm_0", _val(m0).float())
        self.register_buffer("bs_0", torch.sqrt(_val(v0).float() + 1e-8))
        self._has_norm = True

    def _normalized_input(
        self, stage_params: List[torch.Tensor], beam_state_0: torch.Tensor
    ) -> torch.Tensor:
        if getattr(self, "_has_norm", False):
            b0 = (beam_state_0 - self.bm_0.to(beam_state_0.device)) / self.bs_0.to(beam_state_0.device)
            sp = [
                (p - getattr(self, f"pm_{i}").to(p.device)) / getattr(self, f"ps_{i}").to(p.device)
                for i, p in enumerate(stage_params)
            ]
        else:
            b0 = beam_state_0
            sp = stage_params
        return torch.cat([b0, *sp], dim=1)

    def forward(self, stage_params: List[torch.Tensor], beam_state_0: torch.Tensor) -> torch.Tensor:
        """Return (batch,) raw logits -- NOT probabilities.

        Kept as logits (no sigmoid) so training can use BCEWithLogitsLoss
        directly, which is numerically stable near saturation. Use
        predict_proba() for inference.
        """
        x = self._normalized_input(stage_params, beam_state_0)
        return self.net(x).squeeze(-1)

    def predict_proba(self, stage_params: List[torch.Tensor], beam_state_0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(stage_params, beam_state_0))

    # ── Checkpoint I/O (same pattern as ModularMLP.save/load) ────────────────

    _CONFIG_KEYS = ("hidden_sizes", "dropout")

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "model_state_dict": self.state_dict(),
            "model_config": {key: getattr(self, key) for key in self._CONFIG_KEYS},
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "FailureClassifier":
        """Load from checkpoint. Reads architecture config from the file itself."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        norm = ckpt.get("normalization_metadata") or ckpt.get("norm_stats")

        cfg = ckpt.get("model_config", {})
        auto_kwargs = {key: cfg[key] for key in cls._CONFIG_KEYS if key in cfg}
        auto_kwargs.update(kwargs)
        model = cls(norm_stats=norm, **auto_kwargs)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model.to(device)


def derive_failure_labels(Y: torch.Tensor) -> torch.Tensor:
    """Binary label per row (1.0 = all-particles-lost, 0.0 = otherwise),
    derived directly from the final-stage npart_ratio column of a BeamDataset's
    Y tensor -- the same slice score_from_matrix() uses -- via the canonical
    is_all_particles_lost() cliff, not from the scores tensor/ERROR_SCORE
    sentinel.
    """
    npart_idx = BEAM_STATE_FEATURES.index("npart_ratio")
    final_stage = Y[:, -BEAM_STATE_DIM:]
    npart_ratio = final_stage[:, npart_idx]
    threshold = torch.as_tensor(ALL_PARTICLES_LOST_NPART_RATIO, dtype=npart_ratio.dtype)
    return (npart_ratio <= threshold).float()


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss pos_weight = n_negative / n_positive, to counteract
    the class imbalance of failure labels (~12% positive empirically).
    Falls back to 1.0 (no reweighting) if a batch happens to contain only one
    class -- there is nothing meaningful to weight against in that case.
    """
    n_pos = labels.sum()
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0)
    return n_neg / n_pos
