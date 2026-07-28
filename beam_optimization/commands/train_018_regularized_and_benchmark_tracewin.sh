#!/bin/bash
# Train the eight model-free policies on dataset 018 with all three reward
# regularizers, then benchmark the resulting checkpoints only on TraceWin.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

TRAIN_OUTPUT=beam_optimization/results/train/rl/all_018_200k_three_penalties

python -m beam_optimization train_policies \
  --single-surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_018_0.pt \
  --dataset beam_optimization/env/dataset/018/dataset_train.pt \
  --output "$TRAIN_OUTPUT" \
  --rl-steps 200000 \
  --max-ep-steps 20 \
  --hidden 256 256 \
  --seed 42 \
  --n-seeds 1 \
  --eval-every 1000 \
  --eval-episodes 5 \
  --distance-penalty-weight 0.02 \
  --action-penalty-weight 0.02 \
  --score-regression-penalty-weight 1.0 \
  --skip dyna svg \
&& python -m beam_optimization benchmark \
  --policy-only \
  --tracewin-only \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_018_0.pt \
  --dataset beam_optimization/env/dataset/018/dataset_train.pt \
  --sac "$TRAIN_OUTPUT/sac/sac_agent.pt" \
  --td3 "$TRAIN_OUTPUT/td3/td3_agent.pt" \
  --ppo "$TRAIN_OUTPUT/ppo/ppo_agent.pt" \
  --ddpg "$TRAIN_OUTPUT/ddpg/ddpg_agent.pt" \
  --a2c "$TRAIN_OUTPUT/a2c/a2c_agent.pt" \
  --reinforce "$TRAIN_OUTPUT/reinforce/reinforce_agent.pt" \
  --trpo "$TRAIN_OUTPUT/trpo/trpo_agent.pt" \
  --sb3-sac "$TRAIN_OUTPUT/sb3_sac/sb3_sac_agent.zip" \
  --tracewin beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_2/CB_newMRMS_RFQ_Fields_1.ini \
  --tracewin-episodes 5 \
  --max-ep-steps 20 \
  --policy-seed 42 \
  --hidden 256 256 \
  --output beam_optimization/results/benchmark/benchmark_018_three_penalties_tracewin.json
