"""Centralized default paths used by the command line entrypoints."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = PROJECT_ROOT / "env/tracewin_env/dataset/base/dataset_train.pt"
DEFAULT_SURROGATE_DIR = PROJECT_ROOT / "env/surrogate_env/surrogate/models/base"
DEFAULT_SURROGATE_MODEL = DEFAULT_SURROGATE_DIR / "surrogate_0.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs/all"
DEFAULT_BENCHMARK_OUTPUT = PROJECT_ROOT / "results/benchmark.json"
DEFAULT_TRACEWIN_INI = (
    PROJECT_ROOT
    / "env/tracewin_env/tracewin/TraceWin_workspace/condensed_new.ini"
)
