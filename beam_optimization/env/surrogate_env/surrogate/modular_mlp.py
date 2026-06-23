"""
ModularMLP — Neural surrogate of TraceWin for the ADIGE beam line.

Architecture mirrors the physical structure: one sub-network per accelerator
stage. Each stage takes the latent beam representation + the stage parameters
and outputs both an updated latent and a beam-state prediction at that stage.

Reference design: SUMO_TROLL/SURROGATE_MODEL/models.py
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from beam_optimization.config.adige import (
    BEAM_STATE_DIM, STAGE_PARAM_SIZES, N_STAGES,
    STAGE_PARAM_KEYS, STAGE_MARKERS,
)


class ModularMLP(nn.Module):
    """Differentiable surrogate mapping (initial_beam_state, parameters) → beam_states.

    Input:
        beam_state_0: (batch, BEAM_STATE_DIM) — initial beam state (stage 0 from dataset)
        stage_params:  list of 11 tensors, each (batch, stage_param_size)

    Output (single_output=False): list of 11 tensors, each (batch, BEAM_STATE_DIM)
    Output (single_output=True): tensor (batch, BEAM_STATE_DIM) — final stage only

    Internal normalization: if norm_stats is provided at construction, inputs
    are normalized and outputs are denormalized automatically.
    """

    def __init__(
        self,
        hidden_sizes: List[int] = (256, 256, 256),
        dropout: float = 0.15,
        latent_dim: int = 64,
        out_hidden: List[int] = (256, 256),
        out_dropout: float = 0.15,
        act: nn.Module = None,
        single_output: bool = False,
        # normalization statistics (from dataset)
        norm_stats: Optional[dict] = None,
    ):
        super().__init__()
        self.single_output = single_output
        self.act = act or nn.ReLU()
        self._norm_stats = norm_stats

        param_sizes = list(STAGE_PARAM_SIZES)   # [1,1,2,1,1,1,4,1,1,1,2]
        n_stages = N_STAGES                      # 11
        beam_dim = BEAM_STATE_DIM                # 9

        def _block(in_dim: int, hidden: List[int], drop: float) -> nn.Sequential:
            layers: List[nn.Module] = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), self.act, nn.Dropout(drop)]
                prev = h
            return nn.Sequential(*layers), prev

        # ── Input net (stage 0): beam_state_0 + params[0] → latent ──────────
        in_block, prev_dim = _block(beam_dim + param_sizes[0], list(hidden_sizes), dropout)
        layers_in = list(in_block) + [nn.Linear(prev_dim, latent_dim), nn.LayerNorm(latent_dim), self.act]
        self.input_net = nn.Sequential(*layers_in)

        # ── Stage nets: latent + params[i] → latent (with residual) ─────────
        self.stage_nets = nn.ModuleList()
        for i in range(1, n_stages):
            blk, prev_dim = _block(latent_dim + param_sizes[i], list(hidden_sizes), dropout)
            layers_s = list(blk) + [nn.Linear(prev_dim, latent_dim), nn.LayerNorm(latent_dim), self.act]
            self.stage_nets.append(nn.Sequential(*layers_s))

        # ── Output nets: latent → beam_state prediction ──────────────────────
        self.output_nets = nn.ModuleList()
        for _ in range(n_stages):
            blk, prev_dim = _block(latent_dim, list(out_hidden), out_dropout)
            layers_o = list(blk) + [nn.Linear(prev_dim, beam_dim)]
            self.output_nets.append(nn.Sequential(*layers_o))

        # Register norm buffers (so they move with .to(device))
        if norm_stats is not None:
            self._register_norm_buffers(norm_stats)

    # ── Normalization helpers ──────────────────────────────────────────────────

    def _register_norm_buffers(self, stats: dict):
        """Register mean/std buffers. Handles both raw-tensor and {"values":tensor} formats."""
        def _val(x):
            return x["values"] if isinstance(x, dict) else x

        for i, (m, v) in enumerate(zip(stats["parameter_means"], stats["parameter_variances"])):
            self.register_buffer(f"pm_{i}", _val(m).float())
            self.register_buffer(f"ps_{i}", torch.sqrt(_val(v).float() + 1e-8))
        for i, (m, v) in enumerate(zip(stats["beam_state_means"], stats["beam_state_variances"])):
            self.register_buffer(f"bm_{i}", _val(m).float())
            self.register_buffer(f"bs_{i}", torch.sqrt(_val(v).float() + 1e-8))
        self._has_norm = True

    def _norm_params(self, stage_params: List[torch.Tensor]) -> List[torch.Tensor]:
        if not getattr(self, "_has_norm", False):
            return stage_params
        out = []
        for i, p in enumerate(stage_params):
            mean = getattr(self, f"pm_{i}").to(p.device)
            std  = getattr(self, f"ps_{i}").to(p.device)
            out.append((p - mean) / std)
        return out

    def _norm_beam(self, beam: torch.Tensor, stage_idx: int) -> torch.Tensor:
        if not getattr(self, "_has_norm", False):
            return beam
        mean = getattr(self, f"bm_{stage_idx}").to(beam.device)
        std  = getattr(self, f"bs_{stage_idx}").to(beam.device)
        return (beam - mean) / std

    def _denorm_beam(self, beam: torch.Tensor, stage_idx: int) -> torch.Tensor:
        if not getattr(self, "_has_norm", False):
            return beam
        mean = getattr(self, f"bm_{stage_idx}").to(beam.device)
        std  = getattr(self, f"bs_{stage_idx}").to(beam.device)
        return beam * std + mean

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        stage_params: List[torch.Tensor],
        beam_state_0: torch.Tensor,
    ):
        """
        Args:
            stage_params: list of 11 tensors (batch, stage_param_size)
            beam_state_0: (batch, 9) initial beam state (raw, un-normalized)

        Returns:
            If single_output=False: list of 11 tensors (batch, 9) — one per stage
            If single_output=True:  tensor (batch, 9) — final stage only
        """
        sp = self._norm_params(stage_params)
        b0 = self._norm_beam(beam_state_0, stage_idx=0)

        latent = self.input_net(torch.cat([b0, sp[0]], dim=1))
        outputs = [self._denorm_beam(self.output_nets[0](latent), stage_idx=1)]

        for i in range(1, len(sp)):
            residual = latent
            latent = self.stage_nets[i - 1](torch.cat([latent, sp[i]], dim=1)) + residual
            outputs.append(self._denorm_beam(self.output_nets[i](latent), stage_idx=i + 1))

        if self.single_output:
            return outputs[-1]
        return outputs

    # ── Checkpoint I/O ─────────────────────────────────────────────────────────

    def save(self, path: str, extra: Optional[dict] = None):
        payload = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "hidden_sizes": [m.out_features for m in self.input_net if isinstance(m, nn.Linear)][:-1],
                "latent_dim": [m.out_features for m in self.input_net if isinstance(m, nn.Linear)][-1],
            },
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "ModularMLP":
        """Load from checkpoint. Reads architecture config from the file itself.

        Compatible with both beam_optimization and SUMO_TROLL checkpoint formats.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        norm = ckpt.get("normalization_metadata") or ckpt.get("norm_stats")

        # Read architecture from checkpoint (SUMO_TROLL uses different key names)
        cfg = ckpt.get("model_config", {})
        auto_kwargs = {}
        if "hidden_sizes_stage" in cfg:      # SUMO_TROLL format
            auto_kwargs["hidden_sizes"] = cfg["hidden_sizes_stage"]
        elif "hidden_sizes" in cfg:
            auto_kwargs["hidden_sizes"] = cfg["hidden_sizes"]
        if "latent_size" in cfg:
            auto_kwargs["latent_dim"] = cfg["latent_size"]
        elif "latent_dim" in cfg:
            auto_kwargs["latent_dim"] = cfg["latent_dim"]
        if "out_net_hidden_sizes" in cfg:
            auto_kwargs["out_hidden"] = cfg["out_net_hidden_sizes"]

        auto_kwargs.update(kwargs)           # explicit kwargs override auto
        model = cls(norm_stats=norm, **auto_kwargs)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model.to(device)
