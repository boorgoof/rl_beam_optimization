#!/bin/bash
# Run Bayesian Optimization against a trained surrogate to find a candidate
# new default parameter set. Prints the best params found (copy them by hand
# into adige.py's default= fields) and saves convergence/summary/delta plots.
# See: beam_optimization/scripts/bayesian_opt.py, README.md section 4.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization bayesian_opt \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_0.pt \
  --output beam_optimization/results/bayesian_opt.json \
  --n-calls 200 \
  --n-runs 3
  # --dataset defaults to the latest numbered dataset in env/dataset/ (falls
  # back to env/dataset/base/dataset_base.pt); pass --dataset <path> to pin one
