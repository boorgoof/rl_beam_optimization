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

python -m beam_optimization sensitivity \
  --output beam_optimization/results/offline_utility/sensitivity.json \
  --escalation-factor 3.0 \
  --max-iterations 8 \
  --target-score-diff 1.0 \
  --tracewin-particles 10000 \
  --tracewin-particle-key nbr_part1 \
  --timeout 180.0 \
  "$@"
  # --workspace/--tracewin default to DEFAULT_TRACEWIN_INI (mutually exclusive)
  # --calc-dir defaults to sensitivity_calc inside the resolved workspace
  # --seed defaults to unseeded (a fresh random TraceWin seed per parameter)
  # --tracewin-threads defaults to unset (TraceWin's own default thread count)
