#!/bin/bash
# Full benchmark on the surrogate: Bayesian optimization + SVG training,
# followed by 50 deterministic-evaluation episodes for the trained custom SAC.
# Requires surrogate_004_0.pt and a completed SAC training run.
# See: beam_optimization/scripts/benchmark.py, README.md section 4 ("benchmark").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization benchmark \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_004_0.pt \
  --dataset beam_optimization/env/dataset/004/dataset_all.pt \
  --sac beam_optimization/runs/all/sac/sac_agent.pt \
  --n-runs 3 \
  --eval-budget 200 \
  --svg-episodes 500 \
  --policy-episodes 50 \
  --max-ep-steps 20 \
  --output beam_optimization/results/benchmark_surrogate.json
