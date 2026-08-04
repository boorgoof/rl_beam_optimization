#!/usr/bin/env bash
# Cold-start Bayesian Optimization: 64 Sobol points followed by 100
# Gaussian-Process-guided TraceWin evaluations. No dataset is loaded.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

source beam_optimization/.venv/bin/activate

python -m beam_optimization bayesian_opt_cold_start \
  --initial-points 64 \
  --guided-calls 100 \
  --bounds-scale 0.6 \
  --tracewin-particles 10000 \
  --timeout 180.0 \
  --retries 2 \
  --output beam_optimization/results/bayesian_opt/bayesian_opt_cold_start_018.json \
  --samples-output beam_optimization/results/bayesian_opt/bayesian_opt_cold_start_018_samples.pt \
  "$@"
  # --workspace/--tracewin default to DEFAULT_TRACEWIN_INI (mutually exclusive)
  # --calc-dir defaults to a dedicated calc folder inside the resolved workspace
  # --bounds-scale mirrors BAYESIAN_SCALE in adige.py; keep them in sync if you recalibrate
  # --seed defaults to a fresh random seed once, then reused from --output if it exists
  # --tracewin-seed-base defaults to unset (no random_seed passed to TraceWin)
  # --tracewin-threads defaults to unset (TraceWin's own default thread count)
