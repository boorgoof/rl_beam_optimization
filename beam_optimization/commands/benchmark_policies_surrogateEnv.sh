#!/usr/bin/env bash
# Benchmark all promoted checkpoints from the final three-seed training run.
# This evaluates policies only; it does not retrain SVG or Bayesian Optimization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

source beam_optimization/.venv/bin/activate

python -m beam_optimization benchmark \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base_018_final/surrogate_018_0.pt \
  --dataset beam_optimization/env/dataset/018/dataset_all.pt \
  --policy-only \
  --sac beam_optimization/results/train/rl/final_thesis_018_a030_s050/sac/sac_agent.zip \
  --ppo beam_optimization/results/train/rl/final_thesis_018_a030_s050/ppo/ppo_agent.zip \
  --td3 beam_optimization/results/train/rl/final_thesis_018_a030_s050/td3/td3_agent.zip \
  --ddpg beam_optimization/results/train/rl/final_thesis_018_a030_s050/ddpg/ddpg_agent.zip \
  --a2c beam_optimization/results/train/rl/final_thesis_018_a030_s050/a2c/a2c_agent.zip \
  --mbpo beam_optimization/results/train/rl/final_thesis_018_a030_s050/dyna/dyna_agent.pt \
  --svg-final beam_optimization/results/train/rl/final_thesis_018_a030_s050/svg_finale/svg_agent.pt \
  --svg-uniform beam_optimization/results/train/rl/final_thesis_018_a030_s050/svg_uniform/svg_agent.pt \
  --policy-episodes 50 \
  --max-ep-steps 20 \
  --policy-seed 42 \
  --hidden 256 256 \
  --output beam_optimization/results/benchmark/final_thesis_018_a030_s050.json \
  "$@"
