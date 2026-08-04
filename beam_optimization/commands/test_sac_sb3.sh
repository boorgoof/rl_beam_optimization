#!/usr/bin/env bash
# Test the candidate Stable Baselines3 SAC checkpoint on TraceWin.
# Modify checkpoint, workspace, seed, and output paths directly below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

python -m beam_optimization test \
  --algo sac \
  --policy beam_optimization/results/train/rl/sb3_sac_smooth050_action030_seed42/sac/sac_agent.zip \
  --env tracewin \
  --tracewin-project beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace/CB_newMRMS_RFQ_Fields_1.ini \
  --calc-dir beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace/tracewin_calc_test \
  --tracewin-timeout 120.0 \
  --max-ep-steps 20 \
  --hidden 256 256 \
  --seed 42 \
  --reset-scale test \
  --output beam_optimization/results/test_RL/sac_sb3/test.json \
  --render \
  --render-dir beam_optimization/results/test_RL/sac_sb3/renders \
  --render-every 1 \
  --dpi 130 \
  --episode-video \
  --episode-video-fps 2 \
  --tracewin-phase-space \
  --max-particles 40000 \
  --bins 150 \
  "$@"

# Optional:
# --deterministic-reset
# --no-episode-video
# --no-tracewin-phase-space
