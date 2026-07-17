#!/bin/bash
# Cold-start Bayesian Optimization: 64 Sobol points followed by 100
# Gaussian-Process-guided TraceWin evaluations. No dataset is loaded.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization bayesian_opt_cold_start \
  --initial-points 64 \
  --guided-calls 100 \
  --output beam_optimization/results/bayesian_opt_cold_start.json \
  --samples-output beam_optimization/results/bayesian_opt_cold_start_samples.pt \
  "$@"
