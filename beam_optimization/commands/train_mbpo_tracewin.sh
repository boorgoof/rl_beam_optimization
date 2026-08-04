#!/usr/bin/env bash
# Train only MBPO with real TraceWin transitions and online surrogate updates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

OUTPUT="beam_optimization/results/train/rl/mbpo_tracewin_018_r270_h30_seed42"
BASE_ENSEMBLE="beam_optimization/results/train/rl/penalty_matrix_018/ensemble_018"
UPDATED_ENSEMBLE="$OUTPUT/updated_ensemble"
UPDATED_DATASET="$OUTPUT/updated_dataset.pt"
DATASET="beam_optimization/env/dataset/018/dataset_train.pt"
TRACEWIN_PROJECT="beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_2/CB_newMRMS_RFQ_Fields_1.ini"

if [ -e "$OUTPUT" ]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
fi

if [ ! -d "$BASE_ENSEMBLE" ]; then
  echo "Missing base surrogate ensemble: $BASE_ENSEMBLE" >&2
  exit 1
fi

if [ ! -f "$DATASET" ]; then
  echo "Missing base dataset: $DATASET" >&2
  exit 1
fi

if [ ! -f "$TRACEWIN_PROJECT" ]; then
  echo "Missing TraceWin project: $TRACEWIN_PROJECT" >&2
  exit 1
fi

python -u -m beam_optimization train_policies \
  --base-ensemble "$BASE_ENSEMBLE" \
  --updated-ensemble "$UPDATED_ENSEMBLE" \
  --dataset "$DATASET" \
  --update-dataset "$UPDATED_DATASET" \
  --output "$OUTPUT" \
  --tracewin "$TRACEWIN_PROJECT" \
  --online-finetune \
  --rl-steps 270 \
  --max-ep-steps 30 \
  --rollout-length 30 \
  --n-synthetic-per-step 40 \
  --mbpo-min-real-samples 21 \
  --model-train-freq 200 \
  --online-mix-ratio 0.25 \
  --hidden 256 256 \
  --seed 42 \
  --n-seeds 1 \
  --distance-penalty-weight 0.02 \
  --action-penalty-weight 0.30 \
  --action-smoothness-penalty-weight 0.50 \
  --score-regression-penalty-weight 5.0 \
  --no-learning-curve \
  --skip \
    sac ppo td3 ddpg a2c \
    sac_custom td3_custom ppo_custom ddpg_custom a2c_custom \
    reinforce_custom trpo_custom \
    svg
