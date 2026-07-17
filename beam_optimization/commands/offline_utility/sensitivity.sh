#!/bin/bash
# Compute single-seed ADIGE parameter sensitivity from TraceWin finite
# differences and save the report under beam_optimization/results/.
# See: beam_optimization/config/offline_utility/sensitivity.py, README.md section 4.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization sensitivity "$@"
