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
  --single-surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_005_0.pt \
  --base-ensemble beam_optimization/env/surrogate_env/surrogate/trained_models/base \
  --updated-ensemble beam_optimization/env/surrogate_env/surrogate/trained_models/updated \
  --output beam_optimization/results/train/rl/all \
  --rl-steps 300000 \
  --svg-episodes 1000 \
  --svg-horizon 20 \
  --rollout-length 1 \
  --max-ep-steps 20 \
  --hidden 256 256 \
  --seed 42 \
  --n-seeds 3 \
  --eval-every 1000 \
  --eval-episodes 5
  # --n-seeds default is 1; the thesis runs use 3 (mean±std learning curves)
  # --single-surrogate pinned: the unpinned default picks the first
  # surrogate_*.pt alphabetically in --base-ensemble, which today is the
  # stale surrogate_001_0.pt (trained on the old, incomplete dataset/001),
  # not the current surrogate_005_0.pt.
  # --dataset defaults to the latest numbered dataset in env/dataset/ (or
  # the next one to be built, if none exist yet); pass --dataset <path> to pin one
  # CAVEAT: --base-ensemble is used whole-folder for the SVG/MBPO ensemble,
  # and still contains that same stale surrogate_001_0.pt alongside
  # surrogate_005_0.pt -- SVG/MBPO will ensemble two models trained on
  # different dataset generations. Clean up trained_models/base before
  # relying on the ensemble uncertainty signal.
  # --quick/--no-learning-curve/--no-tensorboard are off by default (full run,
  # periodic eval, TensorBoard/metrics.csv logging all enabled)
  # --skip defaults to none (all algorithms run)
  # --tracewin defaults to unset (MBPO trains against the surrogate only);
  # --online-finetune/--online-mix-ratio/--update-dataset require --tracewin
