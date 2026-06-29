"""
Default filesystem paths for the beam_optimization package.

CLI scripts use these as defaults; users can override any of them with
explicit arguments. 
"""
from __future__ import annotations
from pathlib import Path

# Absolute path to the package root (the directory that contains this file's
# parent, i.e. .../beam_optimization).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Root folder for flat BeamDataset files.
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "env/dataset"

# Base dataset used to train the base surrogate and to sample initial beam
# states during environment resets.
DEFAULT_BASE_DATASET_DIR = DEFAULT_DATASET_ROOT / "base"
DEFAULT_DATASET = DEFAULT_BASE_DATASET_DIR / "dataset_train.pt"

# Surrogate checkpoint folders. "base" is kept as the clean reference ensemble.
# "updated" is the working ensemble used by default and fine-tuned by online
# TraceWin updates.
DEFAULT_BASE_SURROGATE_DIR = PROJECT_ROOT / "env/surrogate_env/surrogate/models/base"
DEFAULT_UPDATED_SURROGATE_DIR = PROJECT_ROOT / "env/surrogate_env/surrogate/models/updated"


def _has_surrogate_checkpoints(path: Path) -> bool:
    return path.exists() and any(path.glob("surrogate_*.pt"))


# Use the updated working ensemble by default when it exists; otherwise fall
# back to the conserved base ensemble.
DEFAULT_SURROGATE_DIR = (
    DEFAULT_UPDATED_SURROGATE_DIR
    if _has_surrogate_checkpoints(DEFAULT_UPDATED_SURROGATE_DIR)
    else DEFAULT_BASE_SURROGATE_DIR
)

# Single surrogate checkpoint; used by commands that do not need the full ensemble.
DEFAULT_SURROGATE_MODEL = DEFAULT_SURROGATE_DIR / "surrogate_0.pt"

# Root directory where training scripts write RL agent checkpoints and logs.
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs/all"

# JSON file written by the benchmark command.
DEFAULT_BENCHMARK_OUTPUT = PROJECT_ROOT / "results/benchmark.json"

# TraceWin project file used when a command runs the real simulator program 
DEFAULT_TRACEWIN_INI = (
    PROJECT_ROOT
    / "env/tracewin_env/tracewin/TraceWin_workspace/condensed.ini"
)
