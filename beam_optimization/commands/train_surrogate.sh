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

# <-- MODIFY HERE to train on a different dataset (must have dataset_train.pt/dataset_val.pt).
DATASET_DIR="beam_optimization/env/dataset/013"

python -m beam_optimization train_surrogate \
  --train-dataset "$DATASET_DIR/dataset_train.pt" \
  --val-dataset "$DATASET_DIR/dataset_val.pt" \
  --n-surrogates 1 \
  --model-dir beam_optimization/env/surrogate_env/surrogate/trained_models/base \
  --seed 123 \
  --max-epochs 300 \
  --batch-size 256 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 40 \
  --classifier-patience 20 \
  --log-dir beam_optimization/results/train/surrogate
  # --n-surrogates default is 1; the thesis runs use 4
  # --device defaults to unset (trainer auto-selects cuda if available, else cpu)
  # --no-tensorboard is off by default (TensorBoard/metrics.csv logging enabled)
  # --patience stops training early after 40 epochs without val_loss
  # improvement (ReduceLROnPlateau halves the LR first, at 15 epochs of no
  # improvement); pass --patience 0 to always run the full --max-epochs
  # a shared failure_classifier_<dataset>.pt is also trained by default (once
  # per run, regardless of --n-surrogates) to gate score() calls on surrogate
  # predictions near the all-particles-lost cliff; --classifier-patience works
  # like --patience but tracks validation F1 instead of loss; pass
  # --skip-classifier to disable it entirely
