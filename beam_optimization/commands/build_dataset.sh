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
  --target-samples 100
