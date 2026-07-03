#!/bin/bash
# Compare PSO, Bayesian optimization, SVG, and optional RL checkpoints on the surrogate.
# See: beam_optimization/scripts/benchmark.py, README.md section 4 ("benchmark").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization benchmark \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_0.pt \
  --dataset beam_optimization/env/dataset/base/dataset_base.pt \
  --output beam_optimization/runs/benchmark.json \
  --n-runs 3 \
  --eval-budget 3000 \
  --svg-episodes 500
