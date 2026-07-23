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
  --base-ensemble beam_optimization/env/surrogate_env/surrogate/trained_models/base \
  --output beam_optimization/results/train/rl/all \
  --rl-steps 200000 \
  --svg-episodes 1000 \
  --seed 42 \
  --n-seeds 3
  # --single-surrogate defaults to the first surrogate_*.pt found in
  # --base-ensemble; pass --single-surrogate <path> to pin one
  # --dataset defaults to the latest numbered dataset in env/dataset/ (or
  # the next one to be built, if none exist yet); pass --dataset <path> to pin one
