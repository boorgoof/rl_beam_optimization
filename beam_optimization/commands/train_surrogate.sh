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
  --n-surrogates 1
  # --train-dataset/--val-dataset default to the latest numbered dataset's
  # train/val splits in env/dataset/ (e.g. env/dataset/001/dataset_train.pt);
  # pass --train-dataset/--val-dataset <path> to pin a specific one
