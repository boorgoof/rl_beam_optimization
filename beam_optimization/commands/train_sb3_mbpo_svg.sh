#!/usr/bin/env bash
# Train all SB3 algorithms, then MBPO and both SVG variants.
# CUSTOM model-free agents are skipped.
# IMPORTANT: --base-ensemble must contain only surrogate models trained on
# the same dataset and with the same configuration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

OUTPUT="beam_optimization/results/train/rl/final_thesis_018_a030_s050"

if [ -e "$OUTPUT" ]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
fi

python -m beam_optimization train_policies \
  --single-surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base_018_final/surrogate_018_0.pt \
  --base-ensemble beam_optimization/env/surrogate_env/surrogate/trained_models/base_018_final \
  --updated-ensemble beam_optimization/env/surrogate_env/surrogate/trained_models/updated_018_final \
  --dataset beam_optimization/env/dataset/018/dataset_train.pt \
  --output "$OUTPUT" \
  --rl-steps 200000 \
  --svg-episodes 10000 \
  --svg-horizon 20 \
  --rollout-length 5 \
  --max-ep-steps 20 \
  --hidden 256 256 \
  --seed 42 \
  --n-seeds 3 \
  --eval-every 1000 \
  --eval-episodes 5 \
  --distance-penalty-weight 0.02 \
  --action-penalty-weight 0.30 \
  --action-smoothness-penalty-weight 0.50 \
  --score-regression-penalty-weight 5.0 \
  --skip \
    sac_custom td3_custom ppo_custom ddpg_custom a2c_custom \
    reinforce_custom trpo_custom \
  "$@"

# Other available switches:
# --quick
# --no-learning-curve
# --no-tensorboard
#
# MBPO with real TraceWin and online model update (not used by this template):
# --tracewin path/to/CB_newMRMS_RFQ_Fields_1.ini
# --online-finetune
# --online-mix-ratio 0.5
# --update-dataset path/to/updated_dataset.pt
