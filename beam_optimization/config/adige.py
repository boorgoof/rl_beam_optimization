"""
ADIGE accelerator configuration for the PIAVE complex at LNL-INFN.

All physical constants, parameter specs, stage layout, and score function
are defined here. This is the single source of truth for the problem geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

# ── Beam state ────────────────────────────────────────────────────────────────

BEAM_STATE_VARS: Tuple[str, ...] = (
    "npart_ratio", "x0", "y0", "SizeX", "SizeY", "ex", "ey", "x'0", "y'0"
)
BEAM_STATE_DIM: int = len(BEAM_STATE_VARS)          # 9
_BS_IDX: Dict[str, int] = {v: i for i, v in enumerate(BEAM_STATE_VARS)}

# ── Parameter specs ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParameterSpec:
    name: str         # human-readable label, e.g. "AD.SO.01"
    key: str          # TraceWin element key, e.g. "ele[2][5]"
    marker: int       # lattice element index where this param is applied
    default: float    # physical default value
    sensitivity: float  # scale for action / exploration noise


PARAMETERS: Tuple[ParameterSpec, ...] = (
    ParameterSpec("AD.SO.01",   "ele[2][5]",  marker=2,  default=0.365663,   sensitivity=0.00314648863100745),
    ParameterSpec("AD.SO.02",   "ele[4][5]",  marker=4,  default=0.168963,   sensitivity=0.005137432584890818),
    ParameterSpec("AD.ST.04.X", "ele[10][1]", marker=10, default=0.0,        sensitivity=3.0385670293505896e-05),
    ParameterSpec("AD.ST.04.Y", "ele[10][2]", marker=10, default=0.0,        sensitivity=5.068794163336619e-05),
    ParameterSpec("AD.1EQ.01",  "ele[12][2]", marker=12, default=-145.835,   sensitivity=65.84230183803848),
    ParameterSpec("AD.1EQ.02",  "ele[16][2]", marker=16, default=-106.29,    sensitivity=67.28510961882941),
    ParameterSpec("AD.D.02",    "ele[18][5]", marker=18, default=-0.0461848, sensitivity=3.287536401051671e-05),
    ParameterSpec("AD.EM.6",    "ele[21][6]", marker=24, default=-213.9,     sensitivity=191.8965577011313),
    ParameterSpec("AD.EM.8",    "ele[22][6]", marker=24, default=-16.8,      sensitivity=849.5274110435529),
    ParameterSpec("AD.EM.10",   "ele[23][6]", marker=24, default=-1.67,      sensitivity=1781.5965052952715),
    ParameterSpec("AD.EM.12",   "ele[24][6]", marker=24, default=-86.9,      sensitivity=5919.333952109926),
    ParameterSpec("AD.D.03",    "ele[26][5]", marker=26, default=0.0461848,  sensitivity=2.155827279609473e-05),
    ParameterSpec("AD.1EQ.03",  "ele[30][2]", marker=30, default=-192.5536,  sensitivity=48.479287770694626),
    ParameterSpec("AD.1EQ.04",  "ele[35][2]", marker=35, default=0.0,        sensitivity=114.04949113093213),
    ParameterSpec("AD.ST.05.X", "ele[38][1]", marker=38, default=0.0,        sensitivity=8.76029521332319e-05),
    ParameterSpec("AD.ST.05.Y", "ele[38][2]", marker=38, default=0.0,        sensitivity=7.928994938991538e-05),
)

N_PARAMS: int = len(PARAMETERS)  # 16

# Lattice markers where beam state is measured (stage 0 = initial, stages 1-11 = after each element group)
STAGE_MARKERS: Tuple[int, ...] = (0, 2, 4, 10, 12, 16, 18, 24, 26, 30, 35, 38)
N_STAGES: int = len(STAGE_MARKERS) - 1   # 11 parameter stages (not counting initial)
N_BEAM_STATE_STAGES: int = len(STAGE_MARKERS)  # 12 measurement points

INITIAL_NPART: int = 10_000

# ── Stage parameter grouping (for ModularMLP) ─────────────────────────────────
# Each surrogate stage processes the parameters that first appear at that marker.

def _build_stage_layout() -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[int, ...]]:
    from collections import OrderedDict
    stage_keys: OrderedDict[int, List[str]] = OrderedDict()
    for p in PARAMETERS:
        stage_keys.setdefault(p.marker, []).append(p.key)
    keys = tuple(tuple(v) for v in stage_keys.values())
    sizes = tuple(len(k) for k in keys)
    return keys, sizes


STAGE_PARAM_KEYS: Tuple[Tuple[str, ...], ...]  # 11 tuples
STAGE_PARAM_SIZES: Tuple[int, ...]              # (1,1,2,1,1,1,4,1,1,1,2) → sum=16
STAGE_PARAM_KEYS, STAGE_PARAM_SIZES = _build_stage_layout()

# Flat ordered list of all 16 TraceWin keys (used to build param vectors)
PARAM_KEYS: Tuple[str, ...] = tuple(p.key for p in PARAMETERS)

# ── Derived helpers ───────────────────────────────────────────────────────────

def default_params() -> Dict[str, float]:
    """Return {key: default_value} for all 16 parameters."""
    return {p.key: p.default for p in PARAMETERS}


def sensitivity_vec() -> np.ndarray:
    """Return sensitivity as a (16,) numpy array in PARAM_KEYS order."""
    return np.array([p.sensitivity for p in PARAMETERS], dtype=np.float64)


def action_bounds(action_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (low, high) action bounds as (16,) arrays: ±action_scale * sensitivity."""
    s = sensitivity_vec() * action_scale
    return -s.astype(np.float32), s.astype(np.float32)


def params_to_vec(params: Dict[str, float]) -> np.ndarray:
    """Convert {key: value} dict to a (16,) float32 array in PARAM_KEYS order."""
    return np.array([params[k] for k in PARAM_KEYS], dtype=np.float32)


def vec_to_params(vec: np.ndarray) -> Dict[str, float]:
    """Convert a (16,) array back to {key: value} dict."""
    return {k: float(v) for k, v in zip(PARAM_KEYS, vec)}


def params_to_stage_tensors(params: Dict[str, float], device=None) -> List[torch.Tensor]:
    """Convert flat param dict to the list of stage tensors expected by ModularMLP.

    Returns 11 tensors of shapes (1, stage_size), one per surrogate stage.
    """
    tensors = []
    for stage_keys in STAGE_PARAM_KEYS:
        vals = [[params[k] for k in stage_keys]]
        t = torch.tensor(vals, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        tensors.append(t)
    return tensors


def vec_to_stage_tensors(vec: np.ndarray, device=None) -> List[torch.Tensor]:
    """Convert a (16,) parameter vector to stage tensors for ModularMLP."""
    return params_to_stage_tensors(vec_to_params(vec), device=device)

# ── Score function ─────────────────────────────────────────────────────────────

def score(beam_state: Dict[str, float]) -> float:
    """Compute beam quality score from a beam-state dict. Higher is better.

    Typical good values: npart_ratio≈1.0, ex/ey≈0.05, others≈0 → score≈95.
    Failed simulations return error_score (e.g. -999.0).
    """
    r     = beam_state["npart_ratio"]
    ex    = beam_state["ex"]
    ey    = beam_state["ey"]
    x0    = beam_state["x0"]
    y0    = beam_state["y0"]
    SizeX = beam_state["SizeX"]
    SizeY = beam_state["SizeY"]
    xp0   = beam_state["x'0"]
    yp0   = beam_state["y'0"]
    return (100.0 * r
            - 20.0 * ex - 20.0 * ey
            - 10.0 * abs(x0) - 10.0 * abs(y0)
            - 10.0 * abs(xp0) - 10.0 * abs(yp0)
            - 0.1 * SizeX - 0.1 * SizeY)


def score_from_vec(beam_vec: np.ndarray) -> float:
    """Score from a (9,) numpy array in BEAM_STATE_VARS order."""
    return score({v: float(beam_vec[i]) for i, v in enumerate(BEAM_STATE_VARS)})


def score_tensor(beam_state: torch.Tensor) -> torch.Tensor:
    """Differentiable score from a (batch, 9) tensor. Used by gradient_opt.py."""
    r     = beam_state[:, _BS_IDX["npart_ratio"]]
    ex    = beam_state[:, _BS_IDX["ex"]]
    ey    = beam_state[:, _BS_IDX["ey"]]
    x0    = beam_state[:, _BS_IDX["x0"]]
    y0    = beam_state[:, _BS_IDX["y0"]]
    SizeX = beam_state[:, _BS_IDX["SizeX"]]
    SizeY = beam_state[:, _BS_IDX["SizeY"]]
    xp0   = beam_state[:, _BS_IDX["x'0"]]
    yp0   = beam_state[:, _BS_IDX["y'0"]]
    return (100.0 * r
            - 20.0 * ex - 20.0 * ey
            - 10.0 * torch.abs(x0) - 10.0 * torch.abs(y0)
            - 10.0 * torch.abs(xp0) - 10.0 * torch.abs(yp0)
            - 0.1 * SizeX - 0.1 * SizeY)
