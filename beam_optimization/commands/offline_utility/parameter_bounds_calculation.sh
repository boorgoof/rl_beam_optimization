#!/usr/bin/env bash
# Calculate TraceWin parameter bounds and save the JSON report under results/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

source beam_optimization/.venv/bin/activate

python -m beam_optimization parameter_bounds_calculation \
  --tracewin-particles 10000 \
  --tracewin-threads 1 \
  --timeout 180.0 \
  --retries 0 \
  --initial-step-factor 5.0 \
  --outside-step-factor 1.0 \
  --growth-factor 2.0 \
  --tolerance-factor 0.1 \
  --max-expansions 16 \
  --max-bisections 16 \
  --output beam_optimization/results/offline_utility/parameter_bounds.json \
  "$@"
  # --workspace/--tracewin default to DEFAULT_TRACEWIN_INI (mutually exclusive)
  # --calc-dir defaults to parameter_bounds_calc inside the resolved workspace
  # --seed defaults to unset (TraceWin picks its own random_seed)
