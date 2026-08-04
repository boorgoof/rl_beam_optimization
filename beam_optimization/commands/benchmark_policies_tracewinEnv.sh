#!/usr/bin/env bash
# Validate the promoted checkpoint of every final algorithm on real TraceWin.
# Eight policies x three episodes x twenty steps can take several hours.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

source beam_optimization/.venv/bin/activate

python -m beam_optimization benchmark \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base_018_final/surrogate_018_0.pt \
  --dataset beam_optimization/env/dataset/018/dataset_all.pt \
  --classifier-path beam_optimization/env/surrogate_env/surrogate/trained_models/base_018_final/failure_classifier_018.pt \
  --policy-only \
  --tracewin-only \
  --sac beam_optimization/results/train/rl/final_thesis_018_a030_s050/sac/sac_agent.zip \
  --ppo beam_optimization/results/train/rl/final_thesis_018_a030_s050/ppo/ppo_agent.zip \
  --td3 beam_optimization/results/train/rl/final_thesis_018_a030_s050/td3/td3_agent.zip \
  --ddpg beam_optimization/results/train/rl/final_thesis_018_a030_s050/ddpg/ddpg_agent.zip \
  --a2c beam_optimization/results/train/rl/final_thesis_018_a030_s050/a2c/a2c_agent.zip \
  --mbpo beam_optimization/results/train/rl/final_thesis_018_a030_s050/dyna/dyna_agent.pt \
  --svg-final beam_optimization/results/train/rl/final_thesis_018_a030_s050/svg_finale/svg_agent.pt \
  --svg-uniform beam_optimization/results/train/rl/final_thesis_018_a030_s050/svg_uniform/svg_agent.pt \
  --max-ep-steps 20 \
  --policy-seed 42 \
  --hidden 256 256 \
  --tracewin beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_2/CB_newMRMS_RFQ_Fields_1.ini \
  --tracewin-episodes 10 \
  --output beam_optimization/results/benchmark/final_thesis_018_a030_s050_tracewin.json \
  "$@"
