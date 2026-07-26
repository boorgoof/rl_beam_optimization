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

import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
import textwrap
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
    ParameterSpec("AD.SO.01", "ele[2][5]", marker=2, default=0.4311317906379778, sensitivity=0.0008203680437760797, hw_min=None, hw_max=None),
    #stage 1
    ParameterSpec("AD.SO.02", "ele[29][5]", marker=29, default=0.10111150232736774, sensitivity=0.002867122710046139, hw_min=None, hw_max=None),
    #stage 2
    ParameterSpec("AD.MS.03.X", "ele[38][1]", marker=38, default=4.0706947529191115e-06, sensitivity= 0.00014599749270943364, hw_min=None, hw_max=None),
    ParameterSpec("AD.MS.03.Y", "ele[38][2]", marker=38, default=-7.202734159404677e-06, sensitivity=2.504541673796964e-05, hw_min=None, hw_max=None),
    #stage 3
    ParameterSpec("AD.1EQ.01", "ele[151][2]", marker=151, default=192.30821624378285, sensitivity=23.63581267893679, hw_min=None, hw_max=None),
    #stage 4
    ParameterSpec("AD.MS.04.X", "ele[162][1]", marker=162, default=-6.903900893156021e-06, sensitivity=0.0004512130719117839, hw_min=None, hw_max=None),
    ParameterSpec("AD.MS.04.Y", "ele[162][2]", marker=162, default=1.4283076382417385e-06, sensitivity=0.00015896338985085687, hw_min=None, hw_max=None),
    #stage 5
    ParameterSpec("AD.1EQ.02", "ele[195][2]", marker=195, default=-7.93265212152327, sensitivity=13.912223170558082, hw_min=None, hw_max=None),
    #stage 6
    ParameterSpec("AD.D.02", "ele[197][5]", marker=197, default=-0.04620384302215141, sensitivity=0.00022533311374819292, hw_min=None, hw_max=None),
    #stage 7
    ParameterSpec("AD.EM.6", "ele[200][6]", marker=200, default=-234.60214174611588, sensitivity=10319.947006686732, hw_min=None, hw_max=None),
    ParameterSpec("AD.EM.8", "ele[201][6]", marker=201, default=-1794.14758261054, sensitivity=1050.4430712108588, hw_min=None, hw_max=None),
    ParameterSpec("AD.EM.10", "ele[202][6]", marker=202, default=-595.8945980045671, sensitivity= 597.8835414717153, hw_min=None, hw_max=None),
    ParameterSpec("AD.EM.12", "ele[203][6]", marker=203, default=-1.5967833748777753, sensitivity=1418.9897625338751, hw_min=None, hw_max=None),
    #stage 8
    ParameterSpec("AD.D.03", "ele[205][5]", marker=205, default=0.04620542012403646, sensitivity=1.7508343652815022e-05, hw_min=None, hw_max=None),
    #stage 9
    ParameterSpec("AD.1EQ.03", "ele[225][2]", marker=225, default=5.56477802375267, sensitivity=12.689121497919485, hw_min=None, hw_max=None),
    #stage 10
    ParameterSpec("AD.MS.05.X", "ele[261][1]", marker=261, default=-2.146875364017188e-05, sensitivity= 0.0004565309126443308, hw_min=None, hw_max=None),
    ParameterSpec("AD.MS.05.Y", "ele[261][2]", marker=261, default=2.669825715770255e-06, sensitivity= 0.00041901724439086273, hw_min=None, hw_max=None),
    #stage 11
    ParameterSpec("AD.1EQ.04", "ele[280][2]", marker=280, default=-193.180081121261, sensitivity=55.49152576140488, hw_min=None, hw_max=None),
    #stage 12
    
)

# Number of tunable parameters in the ADIGE beam line.
N_PARAMS: int = len(PARAMETERS)

# Lattice markers where the beam state is recorded.
# Stage 0 is the input beam; stages 1..12 are surrogate/TraceWin output stages.
STAGE_MARKERS: Tuple[int, ...] = (
    0, 2, 29, 108, 151, 162, 195, 197, 203, 205, 225, 261, 332,
)  # Markers 108 and 332 replace 38 and 280 (SLIT.04 and AD.BI.05).
N_OUTPUT_STAGES: int = len(STAGE_MARKERS) - 1  # 12 output stages, excluding input stage 0
N_STAGES: int = len(STAGE_MARKERS)             # 13 total stages, including input stage 0

# Stage visibility for RL observations, in STAGE_MARKERS order.
# True means the stage is included in the flattened Gym observation.
# Default observation: beam0 + marker 108 + final marker 332.
OBSERVATION_STAGE_MASK: Tuple[bool, ...] = (
    True,   # stage 0: beam0
    False,  # marker 2
    False,  # marker 29
    True,   # marker 108: slit:04
    False,  # marker 151
    False,  # marker 162
    False,  # marker 195
    False,  # marker 197
    False,  # marker 203
    False,  # marker 205
    False,  # marker 225
    False,  # marker 261
    True,   # marker 332: AD.BI.05
)

# number of particles in the initial beam state (used to compute npart_ratio)
INITIAL_NPART: int = 10_000

# Episode horizon: env steps before truncation (used by all beam envs).
MAX_STEPS: int = 20

# DATASET_SCALE and TRAIN_RESET_SCALE are independently calibrated via
# `exploration_scale_calculation` (offline_utility): each run finds the
# largest Gaussian scale for which BOTH the dataset and Bayesian-Sobol
# distributions reach the given --target-success-rate (TraceWin succeeds AND
# score != ERROR_SCORE). adige.py is never edited automatically -- run the
# calibration and paste the printed value here yourself.
#   DATASET_SCALE:     --target-success-rate 0.80 -- deliberately looser, so
#                       ~20% of the dataset used to train the surrogate also
#                       covers failure/near-boundary behavior.
#   TRAIN_RESET_SCALE: --target-success-rate 0.90 -- training resets should
#                       mostly land in valid, recoverable states.
DATASET_SCALE: float = 0.6
TRAIN_RESET_SCALE: float = 0.45
TEST_RESET_SCALE: float = DATASET_SCALE       # evaluation resets deliberately use the same gaussian width as dataset generation
BAYESIAN_SCALE: float = DATASET_SCALE         # default Bayesian-opt space per parameter is [default - BAYESIAN_SCALE*sensitivity, default + BAYESIAN_SCALE*sensitivity], intersected with hw_min/hw_max

# ACTION_SCALE is a fixed fraction of DATASET_SCALE, not separately
# calibrated: with MAX_STEPS=20, a full directed trajectory covers
# +-2*DATASET_SCALE, i.e. a policy that wants to can move from one side of
# the dataset-generation gaussian to the other.
ACTION_SCALE: float = DATASET_SCALE / (MAX_STEPS/2)  # max per-step RL action, step_max_p = ACTION_SCALE * sensitivity_p

# Score assigned when the final particle fraction is operationally unusable.
ERROR_SCORE: float = -999.0

# npart_ratio at/below this on the final stage means all particles are
# physically lost. Used by score() (and its variants) as the ERROR_SCORE
# cliff -- shared by TraceWin, the surrogate, Bayesian optimization and the
# dataset builder.
ALL_PARTICLES_LOST_NPART_RATIO: float = 0.0

# RL-only episode-termination threshold on the final stage's npart_ratio,
# stricter than ALL_PARTICLES_LOST_NPART_RATIO. score()/ERROR_SCORE (used by
# the Bayesian optimizer and the dataset builder) are unaffected -- a state
# between ALL_PARTICLES_LOST_NPART_RATIO and RL_MIN_NPART_RATIO still gets a
# real, valid physical score, but a training/eval episode still ends there.
# The boundary itself (exactly 10%) remains valid.
RL_MIN_NPART_RATIO: float = 0.10

# RL uses the absolute physical score on valid steps, normalized to a stable
# numerical range.
REWARD_SCORE_SCALE: float = 100.0

# A physical beam-loss transition ends the episode and receives this RL reward.
# ERROR_SCORE remains the separate physical score stored in simulation results.
TERMINAL_FAILURE_REWARD: float = -10.0

# Random Gaussian resets that land directly in a terminal physical state are
# resampled from the same distribution up to this limit.
MAX_TERMINAL_RESET_ATTEMPTS: int = 32

# Beam-quality score weights, shared by score(), score_from_vec() and score_tensor(). 
SCORE_WEIGHTS: Dict[str, float] = {
    "npart_ratio": 100.0,  # primary objective: preserve transmitted particles
    "emittance":    10.0,  # secondary objective: reduce final ex/ey
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
_SCORE_IMPLEMENTATION_FUNCTIONS: Tuple[str, ...] = (
    "score",
    "score_from_vec",
    "score_from_matrix",
    "score_tensor",
)


def _normalized_function_ast(source: str) -> str:
    """Return a formatting/comment-independent AST representation."""
    module = ast.parse(textwrap.dedent(source))
    function = next(
        (
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    if function is None:
        raise ValueError("Source does not contain a function definition")
    if (
        function.body
        and isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
    ):
        function.body = function.body[1:]
    return ast.dump(function, annotate_fields=True, include_attributes=False)


def score_implementation_sha256() -> str:
    """Fingerprint the normalized implementations of every official score API."""
    normalized = []
    for name in _SCORE_IMPLEMENTATION_FUNCTIONS:
        function = globals().get(name)
        if function is None:
            raise RuntimeError(f"Score implementation {name!r} is not defined")
        normalized.append(_normalized_function_ast(inspect.getsource(function)))
    canonical = "\n".join(normalized)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def score_function_metadata() -> Dict[str, object]:
    """Return the complete serializable identity of the active score."""
    metadata: Dict[str, object] = {
        "name": SCORE_FUNCTION_NAME,
        "version": SCORE_FUNCTION_VERSION,
        "formula": (
            "ERROR_SCORE if npart_ratio <= ALL_PARTICLES_LOST_NPART_RATIO; "
            "otherwise linear transmission reward minus emittance, offset, "
            "angle, and size penalties"
        ),
        "beam_state_features": list(BEAM_STATE_FEATURES),
        "all_particles_lost_npart_ratio": float(ALL_PARTICLES_LOST_NPART_RATIO),
        "error_score": float(ERROR_SCORE),
        "weights": {key: float(value) for key, value in SCORE_WEIGHTS.items()},
        "references": {key: float(value) for key, value in SCORE_REFERENCES.items()},
        "implementation_sha256": score_implementation_sha256(),
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
    for idx in observation_stage_indices():
        if idx == 0:
            labels.append("beam0")
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
def is_all_particles_lost(value: object) -> bool:
    """Return whether a scalar npart_ratio means all particles are lost.

    The threshold is represented in the value's NumPy dtype before comparison,
    so the boundary (``0.0``) remains valid when stored as ``float32``.
    """
    ratio = np.asarray(value)
    threshold = np.asarray(ALL_PARTICLES_LOST_NPART_RATIO, dtype=ratio.dtype)
    return bool(ratio <= threshold)


def below_rl_min_npart_ratio(value: object) -> bool:
    """Return whether a scalar ratio is strictly below the RL-only episode-
    termination threshold (RL_MIN_NPART_RATIO), independent of score()'s own
    ALL_PARTICLES_LOST_NPART_RATIO cliff.
    """
    ratio = np.asarray(value)
    threshold = np.asarray(RL_MIN_NPART_RATIO, dtype=ratio.dtype)
    return bool(ratio < threshold)


def score(beam_state: Dict[str, float]) -> float:
    """Compute a scalar beam quality score (at a specific stage) from a beam-state dict. Higher is better.

    Beams with all particles lost (``npart_ratio <= ALL_PARTICLES_LOST_NPART_RATIO``)
    receive ``ERROR_SCORE``.
    A beam exactly at ``SCORE_REFERENCES`` with full transmission scores 100.
    Values better than a reference receive a linear bonus; worse values
    receive a linear penalty.
    """
    if is_all_particles_lost(beam_state["npart_ratio"]):
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
    return score({v: beam_vec[i] for i, v in enumerate(BEAM_STATE_FEATURES)})


def score_from_matrix(beam_vecs: np.ndarray) -> np.ndarray:
    """Vectorized score for an ``(N, 9)`` array in ``BEAM_STATE_FEATURES`` order.

    Row-wise identical to score()/score_from_vec(), including the
    all-particles-lost threshold; used where scoring one row at a time would
    be a hot loop (e.g. recomputing a whole dataset's scores).
    """
    source = np.asarray(beam_vecs)
    threshold = np.asarray(ALL_PARTICLES_LOST_NPART_RATIO, dtype=source.dtype).item()
    arr = source.astype(np.float64, copy=False)
    w = SCORE_WEIGHTS
    ref = SCORE_REFERENCES
    col = lambda name: arr[:, _BS_IDX[name]]
    all_particles_lost = col("npart_ratio") <= threshold
    transmission = np.clip(col("npart_ratio"), 0.0, 1.0)
    regular_score = (w["npart_ratio"] * transmission
                     - w["emittance"] * ((col("ex") - ref["ex"]) + (col("ey") - ref["ey"]))
                     - w["offset"]    * ((np.abs(col("x0")) - ref["x0"]) + (np.abs(col("y0")) - ref["y0"]))
                     - w["angle"]     * ((np.abs(col("x'0")) - ref["x'0"]) + (np.abs(col("y'0")) - ref["y'0"]))
                     - w["size"]      * ((col("SizeX") - ref["SizeX"]) + (col("SizeY") - ref["SizeY"])))
    return np.where(all_particles_lost, ERROR_SCORE, regular_score)


def score_tensor(beam_state: torch.Tensor) -> torch.Tensor:
    """Differentiable score from a (batch, 9) tensor.
    Used by DifferentiableSurrogateEnv (SVG). Same weights and
    all-particles-lost threshold as score().
    """
    w = SCORE_WEIGHTS
    ref = SCORE_REFERENCES
    col = lambda name: beam_state[:, _BS_IDX[name]]
    all_particles_lost = col("npart_ratio") <= ALL_PARTICLES_LOST_NPART_RATIO
    transmission = torch.clamp(col("npart_ratio"), min=0.0, max=1.0)
    regular_score = (w["npart_ratio"] * transmission
                     - w["emittance"] * ((col("ex") - ref["ex"])+ (col("ey") - ref["ey"]))
                     - w["offset"]    * ((torch.abs(col("x0")) - ref["x0"]) + (torch.abs(col("y0")) - ref["y0"]))
                     - w["angle"]     * ((torch.abs(col("x'0")) - ref["x'0"]) + (torch.abs(col("y'0")) - ref["y'0"]))
                     - w["size"]      * ((col("SizeX") - ref["SizeX"]) + (col("SizeY") - ref["SizeY"])))
    return torch.where(
        all_particles_lost,
        regular_score.new_full((), ERROR_SCORE),
        regular_score,
    )
