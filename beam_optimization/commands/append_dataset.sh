#!/usr/bin/env bash
# Append completed datasets 020, 021, and 022 to dataset 018.
# Modify --inputs and --output-dir as needed. Run the same append only once.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

MERGE_LOG="beam_optimization/env/dataset/018/merge_log.json"
if [ -f "$MERGE_LOG" ] \
  && rg -Fq '"path": "beam_optimization/env/dataset/020/dataset_all.pt"' "$MERGE_LOG" \
  && rg -Fq '"path": "beam_optimization/env/dataset/021/dataset_all.pt"' "$MERGE_LOG" \
  && rg -Fq '"path": "beam_optimization/env/dataset/022/dataset_all.pt"' "$MERGE_LOG"
then
  echo "Datasets 020, 021 and 022 are already recorded in $MERGE_LOG; refusing to append them twice." >&2
  exit 1
fi

python -m beam_optimization merge_datasets \
  --inputs \
    beam_optimization/env/dataset/020/dataset_all.pt \
    beam_optimization/env/dataset/021/dataset_all.pt \
    beam_optimization/env/dataset/022/dataset_all.pt \
  --output-dir beam_optimization/env/dataset/018 \
  --append \
  --seed 123 \
  "$@"

# Add --allow-running only if you intentionally want stable snapshots of
# datasets that are still being generated.
