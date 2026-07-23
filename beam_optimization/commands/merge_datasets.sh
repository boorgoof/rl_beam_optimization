#!/bin/bash
# Merge dataset_all.pt files and create fresh 80/10/10 splits.
# Pass --allow-running to take stable snapshots of builds still in progress.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization merge_datasets \
  --inputs \
    beam_optimization/env/dataset/001/dataset_all.pt \
    beam_optimization/env/dataset/002/dataset_all.pt \
    beam_optimization/env/dataset/003/dataset_all.pt \
    beam_optimization/env/dataset/004/dataset_all.pt \
  --output-dir beam_optimization/env/dataset/005 \
  "$@"
