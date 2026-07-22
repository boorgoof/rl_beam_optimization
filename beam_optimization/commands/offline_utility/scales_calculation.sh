#!/bin/bash
# Compute DATASET_SCALE, TRAIN_RESET_SCALE, TEST_RESET_SCALE and ACTION_SCALE.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization scales_calculation "$@"
