#!/bin/bash
# Full thesis training run: all model-free and model-based RL algorithms on
# SurrogateEnv, 3 seeds each, learning curves as mean±std across seeds.
# Expected duration: several hours (11 algorithms × 3 seeds × 200k steps).
# See: beam_optimization/scripts/train_policies.py, README.md section 4 ("train_policies").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization train_policies \
  --single-surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_0.pt \
  --base-ensemble beam_optimization/env/surrogate_env/surrogate/trained_models/base \
  --output beam_optimization/runs/all \
  --rl-steps 200000 \
  --svg-episodes 1000 \
  --seed 42 \
  --n-seeds 3
  # --dataset defaults to the latest numbered dataset in env/dataset/ (falls
  # back to env/dataset/base/dataset_base.pt); pass --dataset <path> to pin one
