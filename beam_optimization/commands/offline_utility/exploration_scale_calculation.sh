#!/bin/bash
# Calibrate one shared DATASET_SCALE/BAYESIAN_SCALE from TraceWin success rates.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization exploration_scale_calculation "$@"
