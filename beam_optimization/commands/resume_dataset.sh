#!/usr/bin/env bash
# Resume dataset 022 on TraceWin workspace 2.
# Modify --dataset-dir, --workspace, --target-samples, and --scale as needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

python -u -m beam_optimization build_dataset \
  --target-samples 5000 \
  --dataset-dir beam_optimization/env/dataset/022 \
  --workspace beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_2 \
  --scale 1.0 \
  --timeout 180.0 \
  --retries 2 \
  --retry-sleep 5.0 \
  --no-kill-stale \
  "$@"

# Other available alternatives:
# --tracewin path/to/CB_newMRMS_RFQ_Fields_1.ini   # instead of --workspace
# --calc-dir path/to/tracewin_calc
# --seed 42
# Remove --no-kill-stale when this is the only active TraceWin process.
