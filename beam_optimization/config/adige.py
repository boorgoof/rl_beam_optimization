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
import hashlib
import json
from typing import Dict, List, Tuple

import numpy as np
import torch


#Beam state definitions (used for observations and scoring)
BEAM_STATE_FEATURES: Tuple[str, ...] = (
    "npart_ratio", # fraction of particles surviving to this stage
    "x0",          # beam centroid x position (mm)
    "y0",          # beam centroid y position (mm)
    "SizeX",       # beam size in x direction (mm)
    "SizeY",       # beam size in y direction (mm)
    "ex",          # horizontal emittance (mm.mrad)
    "ey",          # vertical emittance (mm.mrad)
    "x'0",         # beam centroid x angle (mrad)
    "y'0"          # beam centroid y angle (mrad)
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
    sensitivity: float           # parameter delta estimated to change the score by 1 point
    hw_min: float | None = None  # hardware lower bound (from machine specs); None = unknown
    hw_max: float | None = None  # hardware upper bound (from machine specs); None = unknown


# List of all tunable parameters in the ADIGE beam line, in order of appearance in the lattice.
PARAMETERS: Tuple[ParameterSpec, ...] = (
    #stage 0
    ParameterSpec("AD.SO.01", "ele[2][5]", marker=2, default=0.43091585, sensitivity=0.005549377602611649, hw_min=0.3475366924, hw_max=0.4519413622),
    #stage 1
    ParameterSpec("AD.SO.02", "ele[29][5]", marker=29, default=0.1182765, sensitivity=0.0059363552477487555, hw_min=-0.2972244775, hw_max=0.3285595703),
    #stage 2
    ParameterSpec("AD.MS.03.X", "ele[38][1]", marker=38, default=-0.00012662607285091371, sensitivity=2.0955975021403468e-05, hw_min=-293.6584907, hw_max=82.58899437),
    ParameterSpec("AD.MS.03.Y", "ele[38][2]", marker=38, default=-3.095834967129621e-05, sensitivity=2.3068855434515107e-05, hw_min=-0.0005051671516, hw_max=1e-3),
    #stage 3
    ParameterSpec("AD.1EQ.01", "ele[151][2]", marker=151, default=132.58669997411448, sensitivity=40.33872428359653, hw_min=-2901813814, hw_max=2901829649),
    #stage 4
    ParameterSpec("AD.MS.04.X", "ele[162][1]", marker=162, default=-4.985770555661989e-07, sensitivity=2.734295051197709e-05, hw_min=-0.7835099654, hw_max=0.001219838731),
    ParameterSpec("AD.MS.04.Y", "ele[162][2]", marker=162, default=3.927801310496309e-05, sensitivity=2.7853710396400267e-05, hw_min=-4.513392325, hw_max=0.005521526321),
    #stage 5
    ParameterSpec("AD.1EQ.02", "ele[195][2]", marker=195, default=-8.331752297202513, sensitivity=53.17887338560897, hw_min=-207782.4005, hw_max=23348945.5),
    #stage 6
    ParameterSpec("AD.D.02", "ele[197][5]", marker=197, default=-0.0462106962484192, sensitivity=6.716011927162356e-05, hw_min=-0.2146480211, hw_max=-0.0004267941387),
    #stage 7
    ParameterSpec("AD.EM.6", "ele[200][6]", marker=200, default=160.75890180110406, sensitivity=199.51183090829747, hw_min=None, hw_max=None),
    ParameterSpec("AD.EM.8", "ele[201][6]", marker=201, default=0.156565228083753, sensitivity=1403.5598763495286, hw_min=None, hw_max=None),
    ParameterSpec("AD.EM.10", "ele[202][6]", marker=202, default=0.006109266178659367, sensitivity= 17527.363203900797, hw_min=None, hw_max=None),
    ParameterSpec("AD.EM.12", "ele[203][6]", marker=203, default=24.3, sensitivity=76410.23346373696, hw_min=None, hw_max=None),
    #stage 8
    ParameterSpec("AD.D.03", "ele[205][5]", marker=205, default=0.046211343775075256, sensitivity=1.3492910067832189e-05, hw_min=-0.2146480211, hw_max=0.1273470113),
    #stage 9
    ParameterSpec("AD.1EQ.03", "ele[225][2]", marker=225, default=14.410101650089715, sensitivity=26.231936464031754, hw_min=-321944193.8, hw_max=275961434.5),
    #stage 10
    ParameterSpec("AD.MS.05.X", "ele[261][1]", marker=261, default=0.0001043767114014796, sensitivity=0.00011467726055375513, hw_min=-1e-3, hw_max=1e-3),
    ParameterSpec("AD.MS.05.Y", "ele[261][2]", marker=261, default=-7.68117197562703e-05, sensitivity= 2.331049618977306e-05, hw_min=-1e-3, hw_max=1e-3),
    #stage 11
    ParameterSpec("AD.1EQ.04", "ele[280][2]", marker=280, default=-165.18741827497422, sensitivity=36.62833965354004, hw_min=-365899839.5, hw_max=365915562.2),
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
# Default observation: beam0 + marker 162 + final stage.
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

EXPLORATION_SCALE: float = 0.35               # shared dataset/Bayesian exploration scale, calibrated via `exploration_scale_calculation` (see results/offline_utility/exploration_scale.json)
DATASET_SCALE: float = EXPLORATION_SCALE      # dataset gaussian bell width, dataset_std_p = DATASET_SCALE * sensitivity_p
BAYESIAN_SCALE: float = EXPLORATION_SCALE     # Default Bayesian-opt space per parameter is [default - BAYESIAN_SCALE*sensitivity,default + BAYESIAN_SCALE*sensitivity], intersected with hw_min/hw_max.

# TRAIN_RESET_SCALE and ACTION_SCALE are derived from DATASET_SCALE by
# offline_utility/scales_calculation.py (defaults k_sigma_dataset=3, f_reset=0.25, k_sigma=3, max_steps=20):
#   TRAIN_RESET_SCALE = f_reset * k_sigma_dataset * DATASET_SCALE / k_sigma = 0.0875
#   ACTION_SCALE = (1-f_reset) * k_sigma_dataset * DATASET_SCALE / max_steps = 0.039375
# Test/evaluation resets deliberately use the same gaussian width as dataset generation.
# Note: Re-run `scales_calculation` after changing DATASET_SCALE and paste the values here.
TRAIN_RESET_SCALE: float = 8.749999999999998e-02
TEST_RESET_SCALE: float = DATASET_SCALE
ACTION_SCALE: float =  3.937500000000000e-02  # max per-step RL action, step_max_p = ACTION_SCALE * sensitivity_p

# Gaussian reset scale at which at least 90% of TraceWin probes produce one
# of the definitive physical beam-loss failures. It is intentionally unset
# until `fail_scale_calculation` measures it on the current configuration.
ALL_PARTICLE_LOST_SCALE = None

# Score assigned when the final particle fraction is operationally unusable.
ERROR_SCORE: float = -999.0

# The boundary itself (exactly 10%) remains valid.
MIN_NPART_RATIO: float = 0.10

# RL uses the absolute physical score on valid steps, normalized to a stable
# numerical range.
REWARD_SCORE_SCALE: float = 100.0

# Low transmission keeps ERROR_SCORE as its physical score, but receives a
# bounded RL reward. This preserves avoidance without making one exploratory
# failure dominate a complete episode.
LOW_TRANSMISSION_REWARD: float = -1.0

# Fraction of training resets drawn from the wider TEST_RESET_SCALE
# distribution to expose policies deliberately to boundary/recovery states.
TRAIN_RECOVERY_RESET_PROBABILITY: float = 0.15

# Beam-quality score weights, shared by score(), score_from_vec() and score_tensor(). 
SCORE_WEIGHTS: Dict[str, float] = {
    "npart_ratio": 100.0,  # reward for keeping particles
    "emittance":    15.0,  # primary objective: ex/ey variation from the input reference
    "offset":        1.0,  # |centroid| deviation from the 0 mm reference
    "angle":         1.0,  # |angular centroid| deviation from the 0 mrad reference
    "size":          1.0,  # RMS-size variation from the input reference
}

# Reference beam quality from the simulated PARTRAN input (part_rfq.dst).
SCORE_REFERENCES: Dict[str, float] = {
    "ex":     0.05,      # mm.mrad
    "ey":     0.05,      # mm.mrad
    "x0":     0,         # mm, 
    "y0":     0,         # mm,
    "x'0":    0,         # mrad,
    "y'0":    0,         # mrad, 
    "SizeX":  5,         # mm
    "SizeY":  5,         # mm
}

SCORE_FUNCTION_NAME: str = "adige_linear_beam_quality"
SCORE_FUNCTION_VERSION: int = 1


def score_function_metadata() -> Dict[str, object]:
    """Return the serializable score identity stored with each dataset."""
    metadata: Dict[str, object] = {
        "name": SCORE_FUNCTION_NAME,
        "version": SCORE_FUNCTION_VERSION,
        "formula": (
            "ERROR_SCORE if npart_ratio < MIN_NPART_RATIO; otherwise linear "
            "transmission reward minus emittance, offset, angle, and size penalties"
        ),
        "beam_state_features": list(BEAM_STATE_FEATURES),
        "min_npart_ratio": float(MIN_NPART_RATIO),
        "error_score": float(ERROR_SCORE),
        "weights": {key: float(value) for key, value in SCORE_WEIGHTS.items()},
        "references": {key: float(value) for key, value in SCORE_REFERENCES.items()},
    }
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    metadata["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return metadata


# IMPORTANT: ParameterSpec.sensitivity values were calibrated with the former
# score shaping. Re-run the sensitivity/action-scale calibration before using
# this score for a new RL training campaign.
def _build_stage_layout() -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[int, ...]]:
    '''
    Parameter grouping in stages: each output stage s (s = 1..N_OUTPUT_STAGES) is
    the lattice section ending at STAGE_MARKERS[s], so a parameter applied at
    marker m belongs to the first stage whose closing marker is >= m. Parameters
    applied at different markers inside the same section (e.g. the AD.EM.* set
    at markers 200-203, all inside the section closed by marker 203) share one
    stage, keeping the layout aligned 1:1 with the model's output stages.

    Returns:
    keys: parameter keys grouped per output stage, in STAGE_MARKERS order.
    sizes: number of parameters in each output stage.
    '''
    grouped: List[List[str]] = [[] for _ in range(N_OUTPUT_STAGES)]
    closing_markers = STAGE_MARKERS[1:]
    for p in PARAMETERS:
        for stage_index, marker in enumerate(closing_markers):
            if p.marker <= marker:
                grouped[stage_index].append(p.key)
                break
        else:
            raise ValueError(
                f"{p.name} has marker {p.marker} beyond the last stage marker "
                f"{closing_markers[-1]}; add its section to STAGE_MARKERS"
            )
    keys = tuple(tuple(v) for v in grouped)
    sizes = tuple(len(k) for k in keys)
    return keys, sizes


STAGE_PARAM_KEYS: Tuple[Tuple[str, ...], ...]
STAGE_PARAM_SIZES: Tuple[int, ...]
STAGE_PARAM_KEYS, STAGE_PARAM_SIZES = _build_stage_layout()

# One parameter group per output stage is a structural invariant of the whole
# surrogate pipeline (ModularMLP, BeamDataset, params_to_stage_tensors).
assert len(STAGE_PARAM_SIZES) == N_OUTPUT_STAGES, (
    f"stage layout produced {len(STAGE_PARAM_SIZES)} parameter groups for "
    f"{N_OUTPUT_STAGES} output stages"
)
assert sum(STAGE_PARAM_SIZES) == N_PARAMS

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
    """Return flattened beam-feature dimension from OBSERVATION_STAGE_MASK."""
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


def reset_std_vec(reset_scale: float) -> np.ndarray:
    """Return reset Gaussian stddevs for an explicit training/test scale."""
    if reset_scale < 0:
        raise ValueError(f"reset_scale must be >= 0, got {reset_scale}")
    return sensitivity_vec() * float(reset_scale)


def dataset_std_vec() -> np.ndarray:
    """Return dataset Gaussian stddevs: sensitivity * DATASET_SCALE."""
    return sensitivity_vec() * DATASET_SCALE


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
        Returns one tensor per stage in STAGE_PARAM_KEYS, each of shape (1, stage_size).
    
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


# Score function
def score(beam_state: Dict[str, float]) -> float:
    """Compute a scalar beam quality score (at a specific stage) from a beam-state dict. Higher is better.

    Beams below ``MIN_NPART_RATIO`` receive ``ERROR_SCORE``.
    A beam exactly at ``SCORE_REFERENCES`` with full transmission scores 100.
    Values better than a reference receive a linear bonus; worse values
    receive a linear penalty.
    """
    if float(beam_state["npart_ratio"]) < MIN_NPART_RATIO:
        return ERROR_SCORE

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


def score_from_matrix(beam_vecs: np.ndarray) -> np.ndarray:
    """Vectorized score for an ``(N, 9)`` array in ``BEAM_STATE_FEATURES`` order.

    Row-wise identical to score()/score_from_vec(), including the minimum
    particle-ratio threshold; used where scoring one row at a time would be a
    hot loop (e.g. recomputing a whole dataset's scores).
    """
    arr = np.asarray(beam_vecs, dtype=np.float64)
    w = SCORE_WEIGHTS
    ref = SCORE_REFERENCES
    col = lambda name: arr[:, _BS_IDX[name]]
    below_minimum_ratio = col("npart_ratio") < MIN_NPART_RATIO
    transmission = np.clip(col("npart_ratio"), 0.0, 1.0)
    regular_score = (w["npart_ratio"] * transmission
                     - w["emittance"] * ((col("ex") - ref["ex"]) + (col("ey") - ref["ey"]))
                     - w["offset"]    * ((np.abs(col("x0")) - ref["x0"]) + (np.abs(col("y0")) - ref["y0"]))
                     - w["angle"]     * ((np.abs(col("x'0")) - ref["x'0"]) + (np.abs(col("y'0")) - ref["y'0"]))
                     - w["size"]      * ((col("SizeX") - ref["SizeX"]) + (col("SizeY") - ref["SizeY"])))
    return np.where(below_minimum_ratio, ERROR_SCORE, regular_score)


def score_tensor(beam_state: torch.Tensor) -> torch.Tensor:
    """Differentiable score from a (batch, 9) tensor.
    Used by DifferentiableSurrogateEnv (SVG). Same weights and minimum
    particle-ratio threshold as score().
    """
    w = SCORE_WEIGHTS
    ref = SCORE_REFERENCES
    col = lambda name: beam_state[:, _BS_IDX[name]]
    below_minimum_ratio = col("npart_ratio") < MIN_NPART_RATIO
    transmission = torch.clamp(col("npart_ratio"), min=0.0, max=1.0)
    regular_score = (w["npart_ratio"] * transmission
                     - w["emittance"] * ((col("ex") - ref["ex"])+ (col("ey") - ref["ey"]))
                     - w["offset"]    * ((torch.abs(col("x0")) - ref["x0"]) + (torch.abs(col("y0")) - ref["y0"]))
                     - w["angle"]     * ((torch.abs(col("x'0")) - ref["x'0"]) + (torch.abs(col("y'0")) - ref["y'0"]))
                     - w["size"]      * ((col("SizeX") - ref["SizeX"]) + (col("SizeY") - ref["SizeY"])))
    return torch.where(
        below_minimum_ratio,
        regular_score.new_full((), ERROR_SCORE),
        regular_score,
    )
