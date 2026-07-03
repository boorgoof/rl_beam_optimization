#!/bin/bash
# Create a new TraceWin dataset and train new base surrogate checkpoints.
# See: beam_optimization/scripts/setup.py, README.md section 4 ("setup").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization setup \
  --target-samples 100 \
  --n-surrogates 3
