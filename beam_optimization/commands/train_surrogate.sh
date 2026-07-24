#!/bin/bash
# Train new base surrogate checkpoints from an existing train/val dataset
# (e.g. produced by build_dataset.sh).
# See: beam_optimization/scripts/train_surrogate.py, README.md section 4.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization train_surrogate \
  --n-surrogates 1 \
  --model-dir beam_optimization/env/surrogate_env/surrogate/trained_models/base \
  --seed 123 \
  --max-epochs 200 \
  --batch-size 256 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --log-dir beam_optimization/results/train/surrogate
  # --n-surrogates default is 1; the thesis runs use 4
  # --train-dataset/--val-dataset default to the latest numbered dataset's
  # train/val splits in env/dataset/ (e.g. env/dataset/001/dataset_train.pt);
  # pass --train-dataset/--val-dataset <path> to pin a specific one
  # --device defaults to unset (trainer auto-selects cuda if available, else cpu)
  # --no-tensorboard is off by default (TensorBoard/metrics.csv logging enabled)
