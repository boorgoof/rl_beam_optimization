#!/usr/bin/env bash
# Train new base surrogate checkpoints from an existing train/val dataset
# (e.g. produced by build_dataset.sh).
# See: beam_optimization/scripts/train_surrogate.py, README.md section 4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

source beam_optimization/.venv/bin/activate

# <-- MODIFY HERE to train on a different dataset (must have dataset_train.pt/dataset_val.pt).
DATASET_DIR="beam_optimization/env/dataset/018"
MODEL_DIR="beam_optimization/env/surrogate_env/surrogate/trained_models/base_018_final"

if [ -e "$MODEL_DIR" ]; then
  echo "Refusing to overwrite existing model directory: $MODEL_DIR" >&2
  exit 1
fi

python -m beam_optimization train_surrogate \
  --train-dataset "$DATASET_DIR/dataset_train.pt" \
  --val-dataset "$DATASET_DIR/dataset_val.pt" \
  --n-surrogates 4 \
  --model-dir "$MODEL_DIR" \
  --seed 123 \
  --max-epochs 300 \
  --batch-size 256 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 40 \
  --classifier-patience 20 \
  --log-dir beam_optimization/results/train/surrogate_018 \
  "$@"
  # Four models are required for the final MBPO/SVG ensemble.
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
