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
    sensitivity: float           # delta param per 10 score points — base physical scale
    hw_min: float | None = None  # hardware lower bound (from machine specs); None = unknown
    hw_max: float | None = None  # hardware upper bound (from machine specs); None = unknown


# List of all tunable parameters in the ADIGE beam line, in order of appearance in the lattice.
PARAMETERS: Tuple[ParameterSpec, ...] = (
    #stage 0
    ParameterSpec("AD.SO.01", "ele[2][5]", marker=2, default=0.43, sensitivity=1.8317e-03, hw_min=0.3475366924, hw_max=0.4519413622),
    #stage 1
    ParameterSpec("AD.SO.02", "ele[29][5]", marker=29, default=0.1, sensitivity=3.6553e-02, hw_min=-0.2972244775, hw_max=0.3285595703),
    #stage 2
    ParameterSpec("AD.MS.03.X", "ele[38][1]", marker=38, default=0.0, sensitivity=4.4809e-04, hw_min=-293.6584907, hw_max=82.58899437),
    ParameterSpec("AD.MS.03.Y", "ele[38][2]", marker=38, default=0.0, sensitivity=1.2222e-04, hw_min=-0.0005051671516, hw_max=1e-3),
    #stage 3
    ParameterSpec("AD.1EQ.01", "ele[151][2]", marker=151, default=0, sensitivity=3.9366e+03, hw_min=-2901813814, hw_max=2901829649),
    #stage 4
    ParameterSpec("AD.MS.04.X", "ele[162][1]", marker=162, default=0, sensitivity=1.1913e-06, hw_min=-0.7835099654, hw_max=0.001219838731),
    ParameterSpec("AD.MS.04.Y", "ele[162][2]", marker=162, default=0, sensitivity=1.7451e-04, hw_min=-4.513392325, hw_max=0.005521526321),
    #stage 5
    ParameterSpec("AD.1EQ.02", "ele[195][2]", marker=195, default=0, sensitivity=3.5623e+01, hw_min=-207782.4005, hw_max=23348945.5),
    #stage 6
    ParameterSpec("AD.D.02", "ele[197][5]", marker=197, default=-0.0462087, sensitivity=8.7381e-05, hw_min=-0.2146480211, hw_max=-0.0004267941387),
    #stage 7
    ParameterSpec("AD.EM.6", "ele[200][6]", marker=200, default=0, sensitivity=4.3740e+02, hw_min=None, hw_max=None),
    ParameterSpec("AD.EM.8", "ele[201][6]", marker=201, default=0, sensitivity=9.3827e-01, hw_min=None, hw_max=None),
    ParameterSpec("AD.EM.10", "ele[202][6]", marker=202, default=0, sensitivity=1.8000e+00, hw_min=None, hw_max=None),
    ParameterSpec("AD.EM.12", "ele[203][6]", marker=203, default=0, sensitivity=4.8600e+01, hw_min=None, hw_max=None),
    #stage 8
    ParameterSpec("AD.D.03", "ele[205][5]", marker=205, default=0.0462087, sensitivity=3.4672e-05, hw_min=-0.2146480211, hw_max=0.1273470113),
    #stage 9
    ParameterSpec("AD.1EQ.03", "ele[225][2]", marker=225, default=0, sensitivity=4.1821e+01, hw_min=-321944193.8, hw_max=275961434.5),
    #stage 10
    ParameterSpec("AD.MS.05.X", "ele[261][1]", marker=261, default=0, sensitivity=1.0349e-03, hw_min=-1e-3, hw_max=1e-3),
    ParameterSpec("AD.MS.05.Y", "ele[261][2]", marker=261, default=0, sensitivity=5.0741e-04, hw_min=-1e-3, hw_max=1e-3),
    #stage 11
    ParameterSpec("AD.1EQ.04", "ele[280][2]", marker=280, default=-206, sensitivity=1.2746e+02, hw_min=-365899839.5, hw_max=365915562.2),
    #stage 12
    
)

# Number of tunable parameters in the ADIGE beam line.
N_PARAMS: int = len(PARAMETERS)

# Lattice markers where the beam state is recorded.
# Stage 0 is the input beam; stages 1..12 are surrogate/TraceWin output stages.
STAGE_MARKERS: Tuple[int, ...] = (0, 2, 29, 38, 151, 162, 195, 197, 203, 205, 225, 261, 280)
N_OUTPUT_STAGES: int = len(STAGE_MARKERS) - 1  # 12 output stages, excluding input stage 0
N_STAGES: int = len(STAGE_MARKERS)             # 13 total stages, including input stage 0

# Stage visibility for RL observations, in STAGE_MARKERS order.
# True means the stage is included in the flattened Gym observation.
# Default: beam0 + final stage.
OBSERVATION_STAGE_MASK: Tuple[bool, ...] = (
    True,   # stage 0: beam0
    False,  # marker 2
    False,  # marker 29
    False,  # marker 38
    False,  # marker 151
    True,   # marker 162
    False,  # marker 195
    False,  # marker 197
    False,  # marker 203
    False,  # marker 205
    False,  # marker 225
    False,  # marker 261
    True,   # marker 280: final
)

# number of particles in the initial beam state (used to compute npart_ratio)
INITIAL_NPART: int = 10_000

# Episode horizon: env steps before truncation (used by all beam envs).
MAX_STEPS: int = 20

DATASET_SCALE: float = 0   # dataset gaussian bell width, dataset_std_p = DATASET_SCALE * sensitivity_p
RESET_SCALE: float =  2.500000000000000e-02      # episode-reset gaussian width, reset_std_p = RESET_SCALE * sensitivity_p
ACTION_SCALE: float =  4.000000000000001e-03    # max per-step RL action, step_max_p = ACTION_SCALE * sensitivity_p

# Score assigned when a simulation fails (TraceWin error, invalid output).
# Large and negative so the agent learns to avoid failure regions.
ERROR_SCORE: float = -999.0

# Beam-quality score weights, shared by score(), score_from_vec() and score_tensor(). 
SCORE_WEIGHTS: Dict[str, float] = {
    "npart_ratio": 100.0,  # reward for keeping particles
    "emittance":   200.0,  # primary objective: ex/ey variation from the input reference
    "offset":        1.0,  # centroid variation from the +/-1 mm reference
    "angle":         1.0,  # angular-centroid variation from the +/-1 mrad reference
    "size":          1.0,  # RMS-size variation from the input reference
}

# Reference beam quality from the simulated PARTRAN input (part_rfq.dst).
SCORE_REFERENCES: Dict[str, float] = {
    "ex":     0.05,      # mm.mrad
    "ey":     0.05,      # mm.mrad
    "x0":     0,         # mm, 
    "y0":     0,         # mm, |
    "x'0":    0,         # mrad,
    "y'0":    0,         # mrad, 
    "SizeX":  5,         # mm
    "SizeY":  5,         # mm
}

# IMPORTANT: ParameterSpec.sensitivity values were calibrated with the former
# score shaping. Re-run the sensitivity/action-scale calibration before using
# this score for a new RL training campaign.
def _build_stage_layout() -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[int, ...]]:
    '''
    Parameter grouping in stages: some parameters are applied at the same lattice marker, so they are grouped into a single stage for the surrogate model.
    Each stage corresponds to a lattice marker and has a list of parameter keys and a corresponding size (number of parameters in that stage).

    Returns:
    keys: parameter keys grouped by their TraceWin marker.
    sizes: number of parameters in each marker group.
    '''
    from collections import OrderedDict
    stage_keys: OrderedDict[int, List[str]] = OrderedDict()
    for p in PARAMETERS:
        stage_keys.setdefault(p.marker, []).append(p.key)
    keys = tuple(tuple(v) for v in stage_keys.values())
    sizes = tuple(len(k) for k in keys)
    return keys, sizes


STAGE_PARAM_KEYS: Tuple[Tuple[str, ...], ...]
STAGE_PARAM_SIZES: Tuple[int, ...]
STAGE_PARAM_KEYS, STAGE_PARAM_SIZES = _build_stage_layout() 

# Flat ordered list of all configured TraceWin keys.
PARAM_KEYS: Tuple[str, ...] = tuple(p.key for p in PARAMETERS)


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
    """Return {key: default_value} for all configured parameters."""
    return {p.key: p.default for p in PARAMETERS}


def sensitivity_vec() -> np.ndarray:
    """Return sensitivity values as a float64 array, in PARAM_KEYS order.
    
    Example:

        (0.00314648863100745, 0.005137432584890818, ...)
    """

    return np.array([p.sensitivity for p in PARAMETERS], dtype=np.float64)


def action_step_vec() -> np.ndarray:
    """Return max per-step parameter deltas: sensitivity * ACTION_SCALE."""
    return sensitivity_vec() * ACTION_SCALE


def reset_std_vec() -> np.ndarray:
    """Return reset Gaussian stddevs: sensitivity * RESET_SCALE."""
    return sensitivity_vec() * RESET_SCALE


def dataset_std_vec() -> np.ndarray:
    """Return dataset Gaussian stddevs: sensitivity * DATASET_SCALE."""
    return sensitivity_vec() * DATASET_SCALE


def hw_bounds_vec() -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Return (hw_min_vec, hw_max_vec) arrays, or None if any bound is missing.

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
    """Return (low, high) action bounds as arrays: ±action_step_vec().

    Example:

        (-sensitivity_vec() * ACTION_SCALE, +sensitivity_vec() * ACTION_SCALE)
    """
    
    s = action_step_vec()
    return -s.astype(np.float32), s.astype(np.float32)


def params_to_vec(params: Dict[str, float]) -> np.ndarray:
    """Convert a {key: value} parameter dict to a float32 array.
    Values are placed in PARAM_KEYS order 

    Example:

         "{ele[2][5]": 0.365663, "ele[4][5]": 0.168963, ...} ->  (0.365663, 0.168963, ...)
    """
    return np.array([params[k] for k in PARAM_KEYS], dtype=np.float32)


def vec_to_params(vec: np.ndarray) -> Dict[str, float]:
    """Convert a parameter array back to {key: value} dict.
    
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
    """Convert a parameter vector to per-stage tensors for ModularMLP.
    Wraps vec_to_params + params_to_stage_tensors.

    Example:

        (0.365663, 0.168963, 0.0, 0.0, ...) -> [tensor([[0.365663]]), tensor([[0.168963]]), tensor([[0.0, 0.0]]), ...]
    """
    return params_to_stage_tensors(vec_to_params(vec), device=device)


# Score function 
def score(beam_state: Dict[str, float]) -> float:
    """Compute a scalar beam quality score (at a specific stage) from a beam-state dict. Higher is better.

    A beam exactly at ``SCORE_REFERENCES`` with full transmission scores 100.
    Values better than a reference receive a linear bonus; worse values
    receive a linear penalty. Failed simulations get ``ERROR_SCORE`` instead
    of calling this function.
    """
    w = SCORE_WEIGHTS
    ref = SCORE_REFERENCES
    transmission = float(np.clip(float(beam_state["npart_ratio"]), 0.0, 1.0))
    return (w["npart_ratio"] * transmission
            - w["emittance"] * ((beam_state["ex"] - ref["ex"]) + (beam_state["ey"] - ref["ey"]))
            - w["offset"]    * ((abs(beam_state["x0"]) - ref["x0"])  + (abs(beam_state["y0"]) - ref["y0"]))
            - w["angle"]     * ((abs(beam_state["x'0"]) - ref["x'0"]) + (abs(beam_state["y'0"]) - ref["y'0"]))
            - w["size"]      * ((beam_state["SizeX"] - ref["SizeX"]) + (beam_state["SizeY"] - ref["SizeY"])))


def score_from_vec(beam_vec: np.ndarray) -> float:
    """Score a ``(9,)`` NumPy array in ``BEAM_STATE_FEATURES`` order.

    Example:

        A vector exactly at SCORE_REFERENCES with npart_ratio=1 scores 100.
    """
    return score({v: float(beam_vec[i]) for i, v in enumerate(BEAM_STATE_FEATURES)})


def score_tensor(beam_state: torch.Tensor) -> torch.Tensor:
    """Differentiable score from a (batch, 9) tensor.
    Used by DifferentiableSurrogateEnv (SVG). Same weights as score().
    """
    w = SCORE_WEIGHTS
    ref = SCORE_REFERENCES
    col = lambda name: beam_state[:, _BS_IDX[name]]
    transmission = torch.clamp(col("npart_ratio"), min=0.0, max=1.0)
    return (w["npart_ratio"] * transmission
            - w["emittance"] * ((col("ex") - ref["ex"])+ (col("ey") - ref["ey"]))
            - w["offset"]    * ((torch.abs(col("x0")) - ref["x0"]) + (torch.abs(col("y0")) - ref["y0"]))
            - w["angle"]     * ((torch.abs(col("x'0")) - ref["x'0"]) + (torch.abs(col("y'0")) - ref["y'0"]))
            - w["size"]      * ((col("SizeX") - ref["SizeX"]) + (col("SizeY") - ref["SizeY"])))
