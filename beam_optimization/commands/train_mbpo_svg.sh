#!/usr/bin/env bash
# Train only MBPO, SVG-final and SVG-uniform with seed 42.
# This is the surrogate-only MBPO variant: the surrogate remains frozen.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

OUTPUT="beam_optimization/results/train/rl/mbpo_svg_018_a030_s050_seed42"
ENSEMBLE="beam_optimization/results/train/rl/penalty_matrix_018/ensemble_018"

if [ -e "$OUTPUT" ]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
fi

if [ ! -f "$ENSEMBLE/surrogate_018_0.pt" ]; then
  echo "Missing surrogate ensemble: $ENSEMBLE" >&2
  exit 1
fi

python -u -m beam_optimization train_policies \
  --base-ensemble "$ENSEMBLE" \
  --dataset beam_optimization/env/dataset/018/dataset_train.pt \
  --output "$OUTPUT" \
  --rl-steps 200000 \
  --svg-episodes 10000 \
  --svg-horizon 20 \
  --rollout-length 5 \
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
    sac ppo td3 ddpg a2c \
    sac_custom td3_custom ppo_custom ddpg_custom a2c_custom \
    reinforce_custom trpo_custom \
  "$@"
