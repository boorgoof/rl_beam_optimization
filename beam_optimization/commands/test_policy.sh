#!/usr/bin/env bash
# Test one generic trained policy on TraceWin.
# Change --algo, --policy, environment paths, seed, and rendering flags below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

python -m beam_optimization test \
  --algo sac \
  --policy beam_optimization/results/train/rl/final_thesis_018_a030_s050/sac/sac_agent.zip \
  --env tracewin \
  --max-ep-steps 20 \
  --hidden 256 256 \
  --seed 42 \
  --reset-scale test \
  --tracewin-project beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace/CB_newMRMS_RFQ_Fields_1.ini \
  --calc-dir beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace/tracewin_calc_test \
  --tracewin-timeout 120.0 \
  --output beam_optimization/results/test_RL/generic_tracewin/test.json \
  --render \
  --render-dir beam_optimization/results/test_RL/generic_tracewin/renders \
  --render-every 1 \
  --dpi 130 \
  --episode-video \
  --episode-video-fps 2 \
  --tracewin-phase-space \
  --max-particles 40000 \
  --bins 150 \
  "$@"

# Available --algo values:
# sac ppo td3 ddpg a2c
# sac_custom td3_custom ppo_custom ddpg_custom a2c_custom
# reinforce_custom trpo_custom mbpo svg_final svg_uniform
#
# To test on the surrogate, replace the TraceWin section with:
# --env surrogate
# --surrogate path/to/surrogate.pt
# --dataset path/to/dataset_all.pt
#
# Other switches:
# --deterministic-reset
# --reset-scale train
# --no-episode-video
# --no-tracewin-phase-space
