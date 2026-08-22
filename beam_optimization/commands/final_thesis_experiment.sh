#!/usr/bin/env bash
# Final thesis pipeline:
#   1. train SAC, PPO, TD3, DDPG, A2C, SVG-final and SVG-uniform;
#   2. use seeds 42, 43 and 44;
#   3. promote each algorithm's best validation seed;
#   4. benchmark each promoted checkpoint on twenty identical TraceWin resets.
# Change the output name and penalty weights together before launching.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

TRAIN_OUTPUT="beam_optimization/results/train/rl/final_thesis_018_a015_s025_steps300k"
BENCHMARK_OUTPUT="beam_optimization/results/benchmark/final_thesis_018_a015_s025_steps300k_tracewin.json"
MODEL_DIR="beam_optimization/env/surrogate_env/surrogate/trained_models/base"
TRACEWIN_PROJECT="beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_5/CB_newMRMS_RFQ_Fields_1.ini"

for REQUIRED_PATH in \
  beam_optimization/env/dataset/018/dataset_train.pt \
  beam_optimization/env/dataset/018/dataset_all.pt \
  "$MODEL_DIR/surrogate_018_0.pt" \
  "$TRACEWIN_PROJECT"
do
  if [ ! -f "$REQUIRED_PATH" ]; then
    echo "Missing required input: $REQUIRED_PATH" >&2
    exit 1
  fi
done

if [ -e "$TRAIN_OUTPUT" ]; then
  echo "Refusing to overwrite existing training output: $TRAIN_OUTPUT" >&2
  exit 1
fi

if [ -e "$BENCHMARK_OUTPUT" ]; then
  echo "Refusing to overwrite existing benchmark output: $BENCHMARK_OUTPUT" >&2
  exit 1
fi

python -m beam_optimization train_policies \
  --single-surrogate "$MODEL_DIR/surrogate_018_0.pt" \
  --base-ensemble "$MODEL_DIR" \
  --updated-ensemble beam_optimization/env/surrogate_env/surrogate/trained_models/updated_018_final \
  --dataset beam_optimization/env/dataset/018/dataset_train.pt \
  --output "$TRAIN_OUTPUT" \
  --rl-steps 300000 \
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
  --action-penalty-weight 0.15 \
  --action-smoothness-penalty-weight 0.25 \
  --score-regression-penalty-weight 5.0 \
  --skip \
    sac_custom td3_custom ppo_custom ddpg_custom a2c_custom \
    reinforce_custom trpo_custom dyna

for CHECKPOINT in \
  "$TRAIN_OUTPUT/sac/sac_agent.zip" \
  "$TRAIN_OUTPUT/ppo/ppo_agent.zip" \
  "$TRAIN_OUTPUT/td3/td3_agent.zip" \
  "$TRAIN_OUTPUT/ddpg/ddpg_agent.zip" \
  "$TRAIN_OUTPUT/a2c/a2c_agent.zip" \
  "$TRAIN_OUTPUT/svg_final/svg_agent.pt" \
  "$TRAIN_OUTPUT/svg_uniform/svg_agent.pt"
do
  if [ ! -f "$CHECKPOINT" ]; then
    echo "Training finished without expected promoted checkpoint: $CHECKPOINT" >&2
    exit 1
  fi
done

python -m beam_optimization benchmark \
  --surrogate "$MODEL_DIR/surrogate_018_0.pt" \
  --dataset beam_optimization/env/dataset/018/dataset_all.pt \
  --policy-only \
  --tracewin-only \
  --tracewin "$TRACEWIN_PROJECT" \
  --no-kill-stale \
  --tracewin-threads 6 \
  --tracewin-timeout 180 \
  --sac "$TRAIN_OUTPUT/sac/sac_agent.zip" \
  --ppo "$TRAIN_OUTPUT/ppo/ppo_agent.zip" \
  --td3 "$TRAIN_OUTPUT/td3/td3_agent.zip" \
  --ddpg "$TRAIN_OUTPUT/ddpg/ddpg_agent.zip" \
  --a2c "$TRAIN_OUTPUT/a2c/a2c_agent.zip" \
  --svg-final "$TRAIN_OUTPUT/svg_final/svg_agent.pt" \
  --svg-uniform "$TRAIN_OUTPUT/svg_uniform/svg_agent.pt" \
  --tracewin-episodes 20 \
  --max-ep-steps 20 \
  --policy-seed 42 \
  --hidden 256 256 \
  --output "$BENCHMARK_OUTPUT"
