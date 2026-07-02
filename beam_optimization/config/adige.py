"""
ADIGE accelerator configuration for LNL-INFN.

ADIGE (Acceleratore Di Ioni a Grande Carica Esotici) is a beam line at LNL
located between SPES and ALPI. It receives 1+ radioactive ion beams from the
SPES target ion source, increases their charge state via a charge breeder, and
delivers them to ALPI for further acceleration.
 
This module defines physical constants, parameter specs, lattice stage layout,
and the beam quality score. 

Note: A lattice is the sequence of magnetic and electrostatic elements that make up the beam line, 
defining where each element sits and what it does to the beam as it travels through

Note: import this as the single source of truth; do not hardcode any of these values elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


#Beam state definitions (used for observations and scoring)
BEAM_STATE_FEATURES: Tuple[str, ...] = (
    "npart_ratio", "x0", "y0", "SizeX", "SizeY", "ex", "ey", "x'0", "y'0"
)
BEAM_STATE_DIM: int = len(BEAM_STATE_FEATURES) # 9
_BS_IDX: Dict[str, int] = {v: i for i, v in enumerate(BEAM_STATE_FEATURES)}


# Specification of each tunable parameter in the ADIGE beam line. Each parameter is
# associated with a lattice element (TraceWin key) and a lattice marker (element index).
# e.g.: (2) AD.SO.01 : FIELD_MAP 50 609 0 70 0.365663 0 0 0 sol1b
@dataclass(frozen=True)
class ParameterSpec:
    name: str                    # human-readable label, e.g. "AD.SO.01"
    key: str                     # TraceWin element key, e.g. "ele[2][5]=0.365663". The parameter to specificated is in marker 2 at position 5
    marker: int                  # lattice element index where this param is applied
    default: float               # physical default value
    sensitivity: float           # Δparam per unit Δscore — base physical scale
    hw_min: float | None = None  # hardware lower bound (from machine specs); None = unknown
    hw_max: float | None = None  # hardware upper bound (from machine specs); None = unknown
    action_scale_rl: float = 1.0 # per-parameter RL action scale, in sensitivity units
    reset_scale: float = 0.5     # per-parameter reset stddev scale, in sensitivity units


# List of all tunable parameters in the ADIGE beam line, in order of appearance in the lattice.
PARAMETERS: Tuple[ParameterSpec, ...] = (
 ParameterSpec("AD.SO.01", "ele[2][5]", marker=2, default=0.365663, sensitivity=5.980825523552588e-04, hw_min=0.0, hw_max=1.5, action_scale_rl=4.837568678487386e+00, reset_scale=5.000000000000001e-01),
    # AD.SO.02: NO STABLE PLATEAU — keep old value 4.350615777335317e-04
    ParameterSpec("AD.SO.02", "ele[4][5]", marker=4, default=0.168963, sensitivity=4.350615777335317e-04, hw_min=0.0, hw_max=1.5, action_scale_rl=1.486214282224481e+01, reset_scale=4.999999999999906e-01),
    ParameterSpec("AD.ST.04.X", "ele[10][1]", marker=10, default=0.0, sensitivity=1.884023027844582e-05, hw_min=-2000.0, hw_max=2000.0, action_scale_rl=2.460332986888881e+06, reset_scale=5.000000020983005e-01),
    ParameterSpec("AD.ST.04.Y", "ele[10][2]", marker=10, default=0.0, sensitivity=4.993476706525125e-06, hw_min=-2000.0, hw_max=2000.0, action_scale_rl=4.142325893605957e+06, reset_scale=5.000000006315652e-01),
    # AD.1EQ.01: NO STABLE PLATEAU — keep old value 2.239468491689368e+01
    ParameterSpec("AD.1EQ.01", "ele[12][2]", marker=12, default=-145.835, sensitivity=2.239468491689368e+01, hw_min=-5000.0, hw_max=5000.0, action_scale_rl=8.261739478094448e+00, reset_scale=5.000000000000053e-01),
    ParameterSpec("AD.1EQ.02", "ele[16][2]", marker=16, default=-106.29, sensitivity=6.074273756406187e+00, hw_min=-5000.0, hw_max=5000.0, action_scale_rl=2.379506336874284e+01, reset_scale=5.000000000000133e-01),
    ParameterSpec("AD.D.02", "ele[18][5]", marker=18, default=-0.0461848, sensitivity=5.389288454887024e-07, hw_min=None, hw_max=None, action_scale_rl=1.000000000000000e+00, reset_scale=5.000000000000000e-01),
    ParameterSpec("AD.EM.6", "ele[21][6]", marker=24, default=-213.9, sensitivity=3.099807835730481e+00, hw_min=None, hw_max=None, action_scale_rl=1.000000000000000e+00, reset_scale=5.000000000000000e-01),
    # AD.EM.8: NO STABLE PLATEAU — keep old value 8.259334331806062e+01
    ParameterSpec("AD.EM.8", "ele[22][6]", marker=24, default=-16.8, sensitivity=8.259334331806062e+01, hw_min=None, hw_max=None, action_scale_rl=1.000000000000000e+00, reset_scale=5.000000000000000e-01),
    # AD.EM.10: NO STABLE PLATEAU — keep old value 9.283778507845258e+01
    ParameterSpec("AD.EM.10", "ele[23][6]", marker=24, default=-1.67, sensitivity=9.283778507845258e+01, hw_min=None, hw_max=None, action_scale_rl=1.000000000000000e+00, reset_scale=5.000000000000000e-01),
    ParameterSpec("AD.EM.12", "ele[24][6]", marker=24, default=-86.9, sensitivity=7.509926465051485e+02, hw_min=None, hw_max=None, action_scale_rl=1.000000000000000e+00, reset_scale=5.000000000000000e-01),
    ParameterSpec("AD.D.03", "ele[26][5]", marker=26, default=0.0461848, sensitivity=1.355515831815867e-06, hw_min=None, hw_max=None, action_scale_rl=1.000000000000000e+00, reset_scale=5.000000000000000e-01),
    ParameterSpec("AD.1EQ.03", "ele[30][2]", marker=30, default=-192.5536, sensitivity=1.024535998204059e+01, hw_min=-5000.0, hw_max=5000.0, action_scale_rl=7.542729961994288e+00, reset_scale=5.000000000000028e-01),
    # AD.1EQ.04: NO STABLE PLATEAU — keep old value 1.598187376008477e+01
    ParameterSpec("AD.1EQ.04", "ele[35][2]", marker=35, default=0.0, sensitivity=1.598187376008477e+01, hw_min=-5000.0, hw_max=5000.0, action_scale_rl=1.195786267896739e+01, reset_scale=5.000000000000090e-01),
    # AD.ST.05.X: NO STABLE PLATEAU — keep old value 6.532795027279745e-04
    ParameterSpec("AD.ST.05.X", "ele[38][1]", marker=38, default=0.0, sensitivity=6.532795027279745e-04, hw_min=-2000.0, hw_max=2000.0, action_scale_rl=1.177490302479993e+05, reset_scale=4.999999999277465e-01),
    # AD.ST.05.Y: NO STABLE PLATEAU — keep old value 9.270255561071355e-06
    ParameterSpec("AD.ST.05.Y", "ele[38][2]", marker=38, default=0.0, sensitivity=9.270255561071355e-06, hw_min=-2000.0, hw_max=2000.0, action_scale_rl=8.297837715589128e+06, reset_scale=5.000000025253357e-01),
)

# number of tunable parameters in the ADIGE beam line 
N_PARAMS: int = len(PARAMETERS) # 16

# Lattice markers where the beam state is recorded.
# Stage 0 is the input beam; stages 1..11 are surrogate/TraceWin output stages.
STAGE_MARKERS: Tuple[int, ...] = (0, 2, 4, 10, 12, 16, 18, 24, 26, 30, 35, 38)
N_OUTPUT_STAGES: int = len(STAGE_MARKERS) - 1  # 11 output stages, excluding input stage 0
N_STAGES: int = len(STAGE_MARKERS)             # 12 total stages, including input stage 0

# Stage visibility for RL observations, in STAGE_MARKERS order.
# True means the stage is included in the flattened Gym observation.
# Default: beam0 + final stage.
OBSERVATION_STAGE_MASK: Tuple[bool, ...] = (
    True,   # stage 0: beam0
    False,  # marker 2
    False,  # marker 4
    False,  # marker 10
    False,  # marker 12
    False,  # marker 16
    False,  # marker 18
    False,  # marker 24
    False,  # marker 26
    False,  # marker 30
    False,  # marker 35
    True,   # marker 38: final
)

# number of particles in the initial beam state (used to compute npart_ratio)
INITIAL_NPART: int = 10_000


def _build_stage_layout() -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[int, ...]]:
    '''
    Parameter grouping in stages: some parameters are applied at the same lattice marker, so they are grouped into a single stage for the surrogate model.
    Each stage corresponds to a lattice marker and has a list of parameter keys and a corresponding size (number of parameters in that stage).

    Returns:
    keys = (
         ("ele[2][5]",), #stage 1, marker 2
        ("ele[4][5]",), #stage 2, marker 4
        ("ele[10][1]", "ele[10][2]"), #stage 3, marker 10
        ("ele[12][2]",), #stage 4, marker 12
        ("ele[16][2]",), #stage 5, marker 16
        ("ele[18][5]",), #stage 6, marker 18
        ("ele[21][6]", "ele[22][6]", "ele[23][6]", "ele[24][6]"), #stage 7, marker 24
        ("ele[26][5]",), #stage 8, marker 26
        ("ele[30][2]",), #stage 9, marker 30
        ("ele[35][2]",), #stage 10, marker 35
        ("ele[38][1]", "ele[38][2]"), #stage 11, marker 38
    )

    sizes = (1, 1, 2, 1, 1, 1, 4, 1, 1, 1, 2) 
    '''
    from collections import OrderedDict
    stage_keys: OrderedDict[int, List[str]] = OrderedDict()
    for p in PARAMETERS:
        stage_keys.setdefault(p.marker, []).append(p.key)
    keys = tuple(tuple(v) for v in stage_keys.values())
    sizes = tuple(len(k) for k in keys)
    return keys, sizes


STAGE_PARAM_KEYS: Tuple[Tuple[str, ...], ...]  # 11 tuples (("ele[2][5]",), ("ele[4][5]",), ("ele[10][1]", "ele[10][2]"), ("ele[12][2]",), ("ele[16][2]",), ("ele[18][5]",), ("ele[21][6]", "ele[22][6]", "ele[23][6]", "ele[24][6]"), ("ele[26][5]",), ("ele[30][2]",), ("ele[35][2]",), ("ele[38][1]", "ele[38][2]"))
STAGE_PARAM_SIZES: Tuple[int, ...]              # (1,1,2,1,1,1,4,1,1,1,2) → sum=16
STAGE_PARAM_KEYS, STAGE_PARAM_SIZES = _build_stage_layout() 

# Flat ordered list of all 16 TraceWin keys 
PARAM_KEYS: Tuple[str, ...] = tuple(p.key for p in PARAMETERS) # ele[2][5], ele[4][5], ele[10][1], ele[10][2], ele[12][2], ele[16][2], ele[18][5], ele[21][6], ele[22][6], ele[23][6], ele[24][6], ele[26][5], ele[30][2], ele[35][2], ele[38][1], ele[38][2]


# helpers 
def observation_stage_indices() -> Tuple[int, ...]:
    """Return selected observation stage indices from OBSERVATION_STAGE_MASK."""
    if len(OBSERVATION_STAGE_MASK) != N_STAGES:
        raise ValueError(
            "OBSERVATION_STAGE_MASK must have length "
            f"{N_STAGES} (len(STAGE_MARKERS)), got {len(OBSERVATION_STAGE_MASK)}"
        )
    indices = tuple(i for i, visible in enumerate(OBSERVATION_STAGE_MASK) if visible)
    if not indices:
        raise ValueError("OBSERVATION_STAGE_MASK must include at least one True value")
    return indices


def observation_stage_labels() -> Tuple[str, ...]:
    """Return human-readable labels for selected observation stages."""
    labels = []
    final_index = N_STAGES - 1
    for idx in observation_stage_indices():
        if idx == 0:
            labels.append("beam0")
        elif idx == final_index:
            labels.append("final")
        else:
            labels.append(f"marker_{STAGE_MARKERS[idx]}")
    return tuple(labels)


def observation_dim() -> int:
    """Return flattened RL observation dimension from OBSERVATION_STAGE_MASK."""
    return len(observation_stage_indices()) * BEAM_STATE_DIM


def select_observation_stages(stages) -> np.ndarray:
    """Select configured beam stages and flatten them into a float32 observation."""
    arr = np.asarray(stages, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != BEAM_STATE_DIM:
        raise ValueError(
            "stages must have shape (n_stages, BEAM_STATE_DIM), got "
            f"{arr.shape}"
        )
    indices = observation_stage_indices()
    if max(indices) >= arr.shape[0]:
        raise ValueError(
            f"OBSERVATION_STAGE_MASK selects stage {max(indices)} but only "
            f"{arr.shape[0]} stages are available"
        )
    return arr[list(indices)].reshape(-1).astype(np.float32)


def select_observation_stages_tensor(stages: List[torch.Tensor]) -> torch.Tensor:
    """Torch counterpart of select_observation_stages(), preserving autograd."""
    indices = observation_stage_indices()
    if max(indices) >= len(stages):
        raise ValueError(
            f"OBSERVATION_STAGE_MASK selects stage {max(indices)} but only "
            f"{len(stages)} stages are available"
        )

    selected = []
    batch_size = None
    for idx in indices:
        stage = stages[idx]
        if not isinstance(stage, torch.Tensor):
            raise TypeError(
                "select_observation_stages_tensor expects torch.Tensor stages, "
                f"got {type(stage).__name__}"
            )
        if stage.dim() == 1:
            stage = stage.unsqueeze(0)
        if stage.dim() != 2 or stage.shape[1] != BEAM_STATE_DIM:
            raise ValueError(
                "each stage tensor must have shape (batch, BEAM_STATE_DIM), got "
                f"{tuple(stage.shape)}"
            )
        if batch_size is None:
            batch_size = stage.shape[0]
        elif stage.shape[0] != batch_size:
            raise ValueError("all selected stage tensors must have the same batch size")
        selected.append(stage)

    return torch.cat(selected, dim=1)


def default_params() -> Dict[str, float]:
    """Return {key: default_value} for all 16 parameters."""
    return {p.key: p.default for p in PARAMETERS}


def sensitivity_vec() -> np.ndarray:
    """Return the 16 sensitivity values as a float64 array, in PARAM_KEYS order.
    
    Example:

        (0.00314648863100745, 0.005137432584890818, ...)
    """

    return np.array([p.sensitivity for p in PARAMETERS], dtype=np.float64)


def action_scale_rl_vec() -> np.ndarray:
    """Return per-parameter RL action scales, in PARAM_KEYS order."""
    return np.array([p.action_scale_rl for p in PARAMETERS], dtype=np.float64)


def reset_scale_vec() -> np.ndarray:
    """Return per-parameter reset scales, in PARAM_KEYS order."""
    return np.array([p.reset_scale for p in PARAMETERS], dtype=np.float64)


def action_step_vec() -> np.ndarray:
    """Return max per-step parameter deltas: sensitivity * action_scale_rl."""
    return sensitivity_vec() * action_scale_rl_vec()


def reset_std_vec() -> np.ndarray:
    """Return reset Gaussian stddevs: sensitivity * reset_scale."""
    return sensitivity_vec() * reset_scale_vec()


def hw_bounds_vec() -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Return (hw_min_vec, hw_max_vec) as float64 arrays of shape (16,), or None if any bound is missing.

    Returns None for a side if at least one parameter has hw_min (or hw_max) = None.
    Use individual ParameterSpec.hw_min/hw_max when you need per-parameter granularity.

    Example:

        hw_min, hw_max = hw_bounds_vec()
        # hw_min: array([0., 0., -2000., ...])  or None
    """
    mins = [p.hw_min for p in PARAMETERS]
    maxs = [p.hw_max for p in PARAMETERS]
    hw_min = None if any(v is None for v in mins) else np.array(mins, dtype=np.float64)
    hw_max = None if any(v is None for v in maxs) else np.array(maxs, dtype=np.float64)
    return hw_min, hw_max


def clip_params_to_hw(params: Dict[str, float]) -> Dict[str, float]:
    """Return params clipped to known hardware bounds.

    Parameters without hw_min/hw_max are left unchanged.
    """
    clipped = dict(params)
    for p in PARAMETERS:
        value = float(clipped[p.key])
        if p.hw_min is not None:
            value = max(value, float(p.hw_min))
        if p.hw_max is not None:
            value = min(value, float(p.hw_max))
        clipped[p.key] = value
    return clipped


def clip_param_vec_to_hw(vec: np.ndarray) -> np.ndarray:
    """Return a parameter vector clipped to known hardware bounds."""
    arr = np.asarray(vec, dtype=np.float32).copy()
    for i, p in enumerate(PARAMETERS):
        if p.hw_min is not None:
            arr[i] = max(float(arr[i]), float(p.hw_min))
        if p.hw_max is not None:
            arr[i] = min(float(arr[i]), float(p.hw_max))
    return arr


def clip_param_tensor_to_hw(tensor: torch.Tensor) -> torch.Tensor:
    """Torch counterpart of clip_param_vec_to_hw(), preserving autograd."""
    clipped = tensor
    for i, p in enumerate(PARAMETERS):
        value = clipped[..., i]
        changed = False
        if p.hw_min is not None:
            value = torch.clamp(value, min=float(p.hw_min))
            changed = True
        if p.hw_max is not None:
            value = torch.clamp(value, max=float(p.hw_max))
            changed = True
        if changed:
            clipped = clipped.clone()
            clipped[..., i] = value
    return clipped


def action_bounds() -> Tuple[np.ndarray, np.ndarray]:
    """Return (low, high) action bounds as (16,) arrays: ±action_step_vec().
    
    Example:
    
        (-sensitivity_vec() * action_scale_rl_vec(), +sensitivity_vec() * action_scale_rl_vec())
    """
    
    s = action_step_vec()
    return -s.astype(np.float32), s.astype(np.float32)


def params_to_vec(params: Dict[str, float]) -> np.ndarray:
    """Convert a {key: value} parameter dict to a float32 array of shape (16,).
    Values are placed in PARAM_KEYS order 

    Example:

         "{ele[2][5]": 0.365663, "ele[4][5]": 0.168963, ...} ->  (0.365663, 0.168963, ...)
    """
    return np.array([params[k] for k in PARAM_KEYS], dtype=np.float32)


def vec_to_params(vec: np.ndarray) -> Dict[str, float]:
    """Convert a (16,) array back to {key: value} dict.
    
        Example:

            (0.365663, 0.168963, ...) -> {ele[2][5]": 0.365663, "ele[4][5]": 0.168963, ...}
    """
    return {k: float(v) for k, v in zip(PARAM_KEYS, vec)}


def params_to_stage_tensors(params: Dict[str, float], device=None) -> List[torch.Tensor]:
    """Split a flat parameter dict into the per-stage tensors expected by ModularMLP.
        Returns a list of 11 tensors, each of shape (1, stage_size).
    
        Example:

            { "ele[2][5]": 0.365663, "ele[4][5]": 0.168963,"ele[10][1]": 0.0, "ele[10][2]": 0.0, ..}  -> [tensor([[0.365663]]), tensor([[0.168963]]), tensor([[0.0, 0.0]]), ...]
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
    """Convert a (16,) parameter vector to per-stage tensors for ModularMLP.
    Wraps vec_to_params + params_to_stage_tensors.

    Example:

        (0.365663, 0.168963, 0.0, 0.0, ...) -> [tensor([[0.365663]]), tensor([[0.168963]]), tensor([[0.0, 0.0]]), ...]
    """
    return params_to_stage_tensors(vec_to_params(vec), device=device)


# Score function 
def score(beam_state: Dict[str, float]) -> float:
    """Compute a scalar beam quality score (at a specific stage) from a beam-state dict. Higher is better.
    
    {"npart_ratio": 1.0, "x0": 0.20, "y0": -0.07, "SizeX": 11.8,"SizeY": 11.8, "ex": 0.089,"ey": 0.089,"x'0": 0.56,"y'0": -0.16} → 95.3

    score formula:
        +100 * (npart_ratio)       reward for keeping particles
        \- 20 * (ex + ey)          penalise transverse emittance growth
        \- 10 * (|x0| + |y0|)      penalise centroid offset
        \- 10 * (|x'0| + |y'0|)    penalise angular centroid offset
        \- 0.1 * (SizeX + SizeY)   penalise beam size
 
    A well-tuned beam (npart_ratio≈1, ex/ey≈0.05, offsets≈0) scores around 95.
    Failed simulations return a large negative sentinel (e.g. -999.0)
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
    """Score from a (9,) numpy array in BEAM_STATE_FEATURES order.

    Example:

        (1.0, 0.20, -0.07, 11.8, 11.8, 0.089, 0.089, 0.56, -0.16) → 95.3
    """
    return score({v: float(beam_vec[i]) for i, v in enumerate(BEAM_STATE_FEATURES)})


def score_tensor(beam_state: torch.Tensor) -> torch.Tensor:
    """Differentiable score from a (batch, 9) tensor. Used by gradient_opt.py.
    
    Example:

        tensor([ [1.0, 0.20, -0.07, 11.8, 11.8, 0.089, 0.089, 0.56, -0.16],
                   [0.99, 0.15,  0.02, 10.5, 10.2, 0.081, 0.084, 0.40, -0.10]]) → tensor([95.3, 96.8])
    """
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
