#!/usr/bin/env bash
# Train a grid of new SAC penalty/horizon combinations without testing them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

SURROGATE="beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_018_0.pt"
TRAIN_DATASET="beam_optimization/env/dataset/018/dataset_train.pt"
GRID_ROOT="beam_optimization/results/train/rl/sac_penalty_horizon_grid_new"
TRAIN_STEPS=300000
TRAIN_SEED=42

mkdir -p "$GRID_ROOT"

run_configuration() {
  local label="$1"
  local horizon="$2"
  local distance="$3"
  local action="$4"
  local smoothness="$5"
  local regression="$6"
  local output="$GRID_ROOT/$label"

  if [ -f "$output/sac/sac_agent.zip" ]; then
    echo "Training already complete, skipping: $label"
  else
    if [ -e "$output" ]; then
      echo "Incomplete output exists; refusing to overwrite: $output"
      exit 1
    fi
    python -m beam_optimization train_policies \
      --single-surrogate "$SURROGATE" \
      --dataset "$TRAIN_DATASET" \
      --output "$output" \
      --rl-steps "$TRAIN_STEPS" \
      --max-ep-steps "$horizon" \
      --hidden 256 256 \
      --seed "$TRAIN_SEED" \
      --n-seeds 1 \
      --eval-every 1000 \
      --eval-episodes 5 \
      --distance-penalty-weight "$distance" \
      --action-penalty-weight "$action" \
      --action-smoothness-penalty-weight "$smoothness" \
      --score-regression-penalty-weight "$regression" \
      --skip \
        sac_custom td3_custom ppo_custom ddpg_custom a2c_custom \
        reinforce_custom trpo_custom \
        ppo td3 ddpg a2c dyna svg
  fi

}

# Horizon 15: light, reference-like, and conservative configurations.
run_configuration "new_h15_d0015_a010_s015_r3"  15 0.015 0.10 0.15 3.0
run_configuration "new_h15_d0020_a015_s025_r5"  15 0.020 0.15 0.25 5.0
run_configuration "new_h15_d0030_a020_s035_r8"  15 0.030 0.20 0.35 8.0
run_configuration "new_h15_d0010_a008_s010_r2"  15 0.010 0.08 0.10 2.0
run_configuration "new_h15_d0040_a025_s040_r12" 15 0.040 0.25 0.40 12.0

# Horizon 20 receives the largest, but still compact, part of the grid.
run_configuration "new_h20_d0015_a010_s020_r75" 20 0.015 0.10 0.20 7.5
run_configuration "new_h20_d0030_a020_s035_r75" 20 0.030 0.20 0.35 7.5
run_configuration "new_h20_d0010_a015_s025_r5"  20 0.010 0.15 0.25 5.0
run_configuration "new_h20_d0040_a015_s025_r5"  20 0.040 0.15 0.25 5.0
run_configuration "new_h20_d0020_a010_s025_r5"  20 0.020 0.10 0.25 5.0
run_configuration "new_h20_d0020_a020_s025_r5"  20 0.020 0.20 0.25 5.0
run_configuration "new_h20_d0020_a015_s035_r5"  20 0.020 0.15 0.35 5.0
run_configuration "new_h20_d0020_a015_s025_r8"  20 0.020 0.15 0.25 8.0
run_configuration "new_h20_d0005_a015_s025_r5"  20 0.005 0.15 0.25 5.0
run_configuration "new_h20_d0050_a015_s025_r5"  20 0.050 0.15 0.25 5.0
run_configuration "new_h20_d0020_a005_s025_r5"  20 0.020 0.05 0.25 5.0
run_configuration "new_h20_d0020_a025_s025_r5"  20 0.020 0.25 0.25 5.0
run_configuration "new_h20_d0020_a015_s015_r5"  20 0.020 0.15 0.15 5.0
run_configuration "new_h20_d0020_a015_s045_r5"  20 0.020 0.15 0.45 5.0
run_configuration "new_h20_d0020_a015_s025_r12" 20 0.020 0.15 0.25 12.0

# Horizon 25: light, central, and conservative configurations.
run_configuration "new_h25_d0015_a010_s020_r3"  25 0.015 0.10 0.20 3.0
run_configuration "new_h25_d0020_a015_s030_r5"  25 0.020 0.15 0.30 5.0
run_configuration "new_h25_d0030_a020_s035_r8"  25 0.030 0.20 0.35 8.0
run_configuration "new_h25_d0010_a008_s015_r2"  25 0.010 0.08 0.15 2.0
run_configuration "new_h25_d0040_a025_s040_r12" 25 0.040 0.25 0.40 12.0

# Horizon 30: three new alternatives around the existing h30 reference.
run_configuration "new_h30_d0010_a010_s015_r3"   30 0.010 0.10 0.15 3.0
run_configuration "new_h30_d0025_a015_s030_r75"  30 0.025 0.15 0.30 7.5
run_configuration "new_h30_d0040_a020_s040_r10"  30 0.040 0.20 0.40 10.0
run_configuration "new_h30_d0015_a008_s020_r5"   30 0.015 0.08 0.20 5.0
run_configuration "new_h30_d0030_a025_s035_r12"  30 0.030 0.25 0.35 12.0
