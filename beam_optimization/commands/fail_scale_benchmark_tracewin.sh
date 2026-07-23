#!/bin/bash
# Stress-test one trained policy with real TraceWin outside the dataset trust region.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization fail_scale_benchmark "$@"
