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

# TraceWin-generated dataset used to train the surrogate and to sample initial
# beam states during environment resets.
DEFAULT_DATASET = PROJECT_ROOT / "env/tracewin_env/dataset/base/dataset_train.pt"

# Directory that holds the base surrogate ensemble checkpoints.
DEFAULT_SURROGATE_DIR = PROJECT_ROOT / "env/surrogate_env/surrogate/models/base"

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
