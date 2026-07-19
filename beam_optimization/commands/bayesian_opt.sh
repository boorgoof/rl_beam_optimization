#!/bin/bash
# Run Bayesian Optimization directly against TraceWin. The Gaussian Process
# is warm-started from the latest dataset and every new point is evaluated by
# the real TraceWin simulator. A seed sequence is optional.
# See: beam_optimization/scripts/bayesian_opt.py, README.md section 4.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization bayesian_opt \
  --output beam_optimization/results/bayesian_optimization/bayesian_opt.json \
  --n-calls 100 \
  --n-runs 1 \
  "$@"
  # --dataset defaults to the latest numbered dataset in env/dataset/ (or
  # the next one to be built, if none exist yet); pass --dataset <path> to pin one
