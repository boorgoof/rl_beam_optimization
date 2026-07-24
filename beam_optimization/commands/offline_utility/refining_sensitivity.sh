#!/bin/bash
# Refine current ParameterSpec.sensitivity values with real TraceWin probes.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization refining_sensitivity \
  --output beam_optimization/results/offline_utility/refining_sensitivity.json \
  --target-score-diff 1.0 \
  --tolerance 0.10 \
  --max-iterations 8 \
  --max-baseline-attempts 5 \
  --tracewin-particles 10000 \
  --tracewin-particle-key nbr_part1 \
  --timeout 180.0 \
  --retries 2 \
  "$@"
  # --workspace/--tracewin default to DEFAULT_TRACEWIN_INI (mutually exclusive)
  # --calc-dir defaults to a dedicated calc folder inside the resolved workspace
  # --seed defaults to unset (per-parameter seeds are not reproducible across reruns)
  # --tracewin-threads defaults to unset (TraceWin's own default thread count)
