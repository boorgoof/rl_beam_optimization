"""
Default filesystem paths for the beam_optimization package.

CLI scripts use these as defaults; users can override any of them with
explicit arguments.
"""
from __future__ import annotations
import os
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

# Surrogate checkpoint folders. "base" is kept as the clean offline reference
# ensemble. "updated" is the working ensemble fine-tuned by online TraceWin
# updates and is used only when explicitly requested by MBPOWithModelUpdate.
DEFAULT_BASE_SURROGATE_DIR = PROJECT_ROOT / "env/surrogate_env/surrogate/trained_models/base"
DEFAULT_UPDATED_SURROGATE_DIR = PROJECT_ROOT / "env/surrogate_env/surrogate/trained_models/updated"
DEFAULT_SINGLE_SURROGATE_MODEL = DEFAULT_BASE_SURROGATE_DIR / "surrogate_0.pt"

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

# Working directory for TraceWin output files when running TraceWinEnv.
DEFAULT_TRACEWIN_ENV_CALC_DIR = Path("/tmp/tracewin_calc")

# Folder name used for TraceWin calculation files created during dataset setup.
DEFAULT_TRACEWIN_CALC_DIR_NAME = "tracewin_calc"

# Writable matplotlib cache shared by all CLI scripts (headless-safe).
MATPLOTLIB_CACHE_DIR = Path("/tmp/beam_optimization_matplotlib")


def configure_matplotlib_cache() -> None:
    """Point matplotlib at a writable cache dir before the first import."""
    MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))


def default_tracewin_calc_dir(dataset_dir: Path) -> Path:
    """Return the default TraceWin calc directory for a generated dataset."""
    return dataset_dir / DEFAULT_TRACEWIN_CALC_DIR_NAME


def default_eval_calc_dir(project_file: Path) -> Path:
    """Return the default TraceWin calc directory used by scripts/test.py."""
    return project_file.parent / "calc"


# Calc directory and results checkpoint for config/utility/sensitivity.py.
DEFAULT_SENSITIVITY_CALC_DIR = PROJECT_ROOT / "env/tracewin_env/tracewin/sensitivity_calc"
DEFAULT_SENSITIVITY_CHECKPOINT = DEFAULT_SENSITIVITY_CALC_DIR / "sensitivity_results.json"
