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

# Base dataset used by environments/algorithms to sample initial beam states
# and by online model-update workflows as the offline fine-tuning dataset.
DEFAULT_BASE_DATASET_DIR = DEFAULT_DATASET_ROOT / "base"
DEFAULT_BASE_DATASET = DEFAULT_BASE_DATASET_DIR / "dataset_base.pt"

# Legacy name kept for CLI scripts that accept a generic "--dataset" argument.
DEFAULT_DATASET = DEFAULT_BASE_DATASET

# Surrogate checkpoint folders. "base" is kept as the clean offline reference
# ensemble. "updated" is the working ensemble fine-tuned by online TraceWin
# updates and is used only when explicitly requested by MBPOWithModelUpdate.
DEFAULT_BASE_SURROGATE_DIR = PROJECT_ROOT / "env/surrogate_env/surrogate/trained_models/base"
DEFAULT_UPDATED_SURROGATE_DIR = PROJECT_ROOT / "env/surrogate_env/surrogate/trained_models/updated"

# Default surrogate paths are deliberately anchored to the base ensemble.
# Algorithms that need the updated ensemble must ask for it explicitly.
DEFAULT_SURROGATE_DIR = DEFAULT_BASE_SURROGATE_DIR
DEFAULT_SINGLE_SURROGATE_MODEL = DEFAULT_BASE_SURROGATE_DIR / "surrogate_0.pt"

# Legacy name kept for scripts that expect a single default surrogate.
DEFAULT_SURROGATE_MODEL = DEFAULT_SINGLE_SURROGATE_MODEL

# Root directory where training scripts write RL agent checkpoints and logs.
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs/all"
DEFAULT_SURROGATE_LOG_DIR = PROJECT_ROOT / "runs/surrogate"

# JSON file written by the benchmark command.
DEFAULT_BENCHMARK_OUTPUT = PROJECT_ROOT / "results/benchmark.json"

# TraceWin project file used when a command runs the real simulator program 
DEFAULT_TRACEWIN_INI = (
    PROJECT_ROOT
    / "env/tracewin_env/tracewin/TraceWin_workspace/condensed.ini"
)
