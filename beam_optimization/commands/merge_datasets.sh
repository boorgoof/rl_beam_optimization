#!/usr/bin/env bash
# Merge dataset_all.pt files and create fresh 80/10/10 splits.
# Pass --allow-running to take stable snapshots of builds still in progress.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

source beam_optimization/.venv/bin/activate

python -m beam_optimization merge_datasets \
  --inputs \
    beam_optimization/env/dataset/020/dataset_all.pt \
    beam_optimization/env/dataset/021/dataset_all.pt \
    beam_optimization/env/dataset/022/dataset_all.pt \
  --output-dir beam_optimization/env/dataset/024 \
  --seed 123 \
  "$@"
  # --allow-running is off by default (see header comment)
  # --append is off by default (merges into a fresh --output-dir); pass
  # --append to instead append --inputs onto the dataset_all.pt already in
  # --output-dir (which must already exist in that case)
