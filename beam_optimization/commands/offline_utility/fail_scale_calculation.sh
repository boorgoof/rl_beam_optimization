#!/bin/bash
# Measure the Gaussian reset scale with >=90% definitive TraceWin physics failures.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization fail_scale_calculation "$@"
