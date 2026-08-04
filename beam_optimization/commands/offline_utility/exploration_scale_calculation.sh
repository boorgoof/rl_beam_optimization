#!/usr/bin/env bash
# Calibrate DATASET_SCALE and TRAIN_RESET_SCALE from TraceWin success rates.
# Run twice with a different --target-success-rate (0.80 for DATASET_SCALE,
# 0.90 for TRAIN_RESET_SCALE); pass the same --sample-seed to both runs so
# only the target rate differs between them. See README.md section 3.3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

source beam_optimization/.venv/bin/activate

python -m beam_optimization exploration_scale_calculation \
  --start-scale 1 \
  --min-scale 0.05 \
  --scale-step 0.05 \
  --target-success-rate 0.80 \
  --samples-per-distribution 32 \
  --tracewin-particles 10000 \
  --timeout 180.0 \
  --retries 2 \
  --output beam_optimization/results/offline_utility/exploration_scale_dataset_080.json \
  "$@"
  # --workspace/--tracewin default to DEFAULT_TRACEWIN_INI (mutually exclusive)
  # --calc-dir defaults to a dedicated calc folder inside the resolved workspace
  # --sample-seed defaults to unseeded (unset: each run draws a fresh design)
  # --tracewin-threads defaults to unset (TraceWin's own default thread count)
