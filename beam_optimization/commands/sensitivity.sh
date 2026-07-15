#!/bin/bash
# Compute ADIGE parameter sensitivity from TraceWin finite differences.
# Prints a stability table and a copy-paste block for adige.py (sensitivity=);
# does NOT modify adige.py — copy the printed values in by hand.
# See: beam_optimization/config/utility/sensitivity.py, README.md section 4.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization sensitivity
