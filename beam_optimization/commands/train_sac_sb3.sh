#!/usr/bin/env bash
# Train only Stable Baselines3 SAC.
# Change OUTPUT and the four penalty weights before starting a new experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

OUTPUT="beam_optimization/results/train/rl/sb3_sac_smooth050_action030_seed42"

if [ -e "$OUTPUT" ]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
fi

python -m beam_optimization train_policies \
  --single-surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_018_0.pt \
  --dataset beam_optimization/env/dataset/018/dataset_train.pt \
  --output "$OUTPUT" \
  --rl-steps 200000 \
  --max-ep-steps 20 \
  --hidden 256 256 \
  --seed 42 \
  --n-seeds 1 \
  --eval-every 1000 \
  --eval-episodes 5 \
  --distance-penalty-weight 0.02 \
  --action-penalty-weight 0.30 \
  --action-smoothness-penalty-weight 0.50 \
  --score-regression-penalty-weight 5.0 \
  --skip \
    sac_custom td3_custom ppo_custom ddpg_custom a2c_custom \
    reinforce_custom trpo_custom \
    ppo td3 ddpg a2c dyna svg \
  "$@"

# Other available switches:
# --quick
# --no-learning-curve
# --no-tensorboard
