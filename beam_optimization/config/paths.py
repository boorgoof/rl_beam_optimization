"""
Default filesystem paths for the beam_optimization package.

CLI scripts use these as defaults; users can override any of them with
explicit arguments.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Absolute path to the package root (the directory that contains this file's
# parent, i.e. .../beam_optimization).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Root folder for flat BeamDataset files.
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "env/dataset"

def latest_numbered_dataset_dir() -> Optional[Path]:
    """Return the highest-numbered dataset directory under
    DEFAULT_DATASET_ROOT (e.g. env/dataset/003), or None if none exist yet.

    Mirrors the numbering scheme built by
    tracewin_dataset_builder.next_numbered_dataset_dir() (numeric-only
    subdirectory names), without importing it: paths.py has no dependency on
    the env package and shouldn't gain one.
    """
    if not DEFAULT_DATASET_ROOT.exists():
        return None
    numbered = [
        child for child in DEFAULT_DATASET_ROOT.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    if not numbered:
        return None
    return max(numbered, key=lambda p: int(p.name))


def next_numbered_dataset_dir() -> Path:
    """Return the dataset directory the next build_dataset.sh run would
    create (e.g. env/dataset/003 if 001 and 002 already exist).

    Mirrors tracewin_dataset_builder.next_numbered_dataset_dir()'s numbering,
    minus its mkdir side effect and its import (would make config depend on
    env). Used by default_dataset_path() as the "nothing built yet" case, so
    callers get a consistent, forward-looking path instead of a dead
    reference to a hand-maintained "base" dataset.
    """
    latest = latest_numbered_dataset_dir()
    next_idx = int(latest.name) + 1 if latest is not None else 1
    return DEFAULT_DATASET_ROOT / f"{next_idx:03d}"


def default_dataset_path(prefix: str = "all") -> Path:
    """Return the dataset .pt file scripts should default to.

    Resolves to f"dataset_{prefix}.pt" in the most recently built numbered
    dataset directory (e.g. env/dataset/003/dataset_all.pt), so that once a
    fresh dataset is built via build_dataset.sh, every script automatically
    starts using it. If no numbered dataset exists yet (or it doesn't have
    this split), returns the path the *next* build would create instead --
    this won't exist on disk until build_dataset.sh actually runs, so
    callers that need to fail loudly should check .exists() themselves
    (like scripts/check.py does) rather than assume the returned path is
    ready to load.
    """
    latest = latest_numbered_dataset_dir()
    if latest is not None:
        candidate = latest / f"dataset_{prefix}.pt"
        if candidate.exists():
            return candidate
    return next_numbered_dataset_dir() / f"dataset_{prefix}.pt"

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
    / "env/tracewin_env/tracewin/TraceWin_workspace/CB_newMRMS_RFQ_Fields_1.ini"
)

# Parent directory for automatically generated TraceWinEnv calculation folders.
TRACEWIN_ENV_CALC_ROOT = Path("/tmp")

# Folder name used for TraceWin calculation files created during dataset setup.
DEFAULT_TRACEWIN_CALC_DIR_NAME = "tracewin_calc"

# Writable matplotlib cache shared by all CLI scripts (headless-safe).
MATPLOTLIB_CACHE_DIR = Path("/tmp/beam_optimization_matplotlib")


def configure_matplotlib_cache() -> None:
    """Point matplotlib at a writable cache dir before the first import."""
    MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))


def new_tracewin_env_calc_dir() -> Path:
    """Return a unique calculation directory for one TraceWinEnv instance."""
    return TRACEWIN_ENV_CALC_ROOT / f"tracewin_calc_{os.getpid()}_{uuid4().hex}"


def default_tracewin_calc_dir(dataset_dir: Path) -> Path:
    """Return the default TraceWin calc directory for a generated dataset."""
    return dataset_dir / DEFAULT_TRACEWIN_CALC_DIR_NAME


def default_eval_calc_dir(project_file: Path) -> Path:
    """Return the default TraceWin calc directory used by scripts/test.py."""
    return project_file.parent / "calc"


# Calc directory and results checkpoint for config/utility/sensitivity.py.
DEFAULT_SENSITIVITY_CALC_DIR = PROJECT_ROOT / "env/tracewin_env/tracewin/sensitivity_calc"
DEFAULT_SENSITIVITY_CHECKPOINT = DEFAULT_SENSITIVITY_CALC_DIR / "sensitivity_results.json"

