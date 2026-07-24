#!/bin/bash
# Create a new TraceWin dataset (train/val/test/all splits).
# See: beam_optimization/scripts/build_dataset.py, README.md section 4.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization build_dataset \
  --target-samples 100 \
  --dataset-root beam_optimization/env/dataset \
  --timeout 180.0 \
  --retries 2 \
  --retry-sleep 5.0
  # --workspace/--tracewin default to DEFAULT_TRACEWIN_INI (mutually exclusive)
  # --dataset-dir defaults to a fresh numbered directory under --dataset-root;
  # pass an existing one (e.g. env/dataset/004) to resume an interrupted build
  # --calc-dir defaults to tracewin_calc_<dataset> inside the resolved workspace
  # --seed defaults to a fresh random seed, saved in builder_state.json
  # --no-kill-stale is off by default (stale TraceWin processes are killed
  # before each simulation)
