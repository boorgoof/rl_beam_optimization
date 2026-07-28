#!/bin/bash
# Test one trained policy step by step on the real TraceWin environment,
# saving per-step renders (parameters/state/score + phase-space) and the
# end-of-episode trend videos (params/state/score/phase-space GIFs).
# Requires the local TraceWin setup described in README.md section 2
# (TraceWin_workspace, licensed binary, SSH launcher).
# See: beam_optimization/scripts/test.py, README.md section 4 ("test").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization test \
  --algo sac \
  --policy beam_optimization/results/train/rl/all/sac/sac_agent.pt \
  --env tracewin \
  --tracewin-project beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace/CB_newMRMS_RFQ_Fields_1.ini \
  --calc-dir beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace/tracewin_calc_test \
  --tracewin-timeout 120.0 \
  --max-ep-steps 20 \
  --hidden 256 256 \
  --seed 42 \
  --reset-scale test \
  --deterministic-reset \
  --output beam_optimization/results/test_RL/test.json \
  --render \
  --render-dir beam_optimization/results/test_RL/renders \
  --render-every 1 \
  --dpi 130 \
  --episode-video \
  --episode-video-fps 2 \
  --tracewin-phase-space \
  --max-particles 40000 \
  --bins 150
  # --calc-dir avoids TraceWin_workspace/calc, currently owned by comunian
  # (0755) from a stale run: almalinux can't chmod/clean it. Fix that
  # directory (sudo -u comunian rm -rf .../calc/*) if you want to drop
  # --calc-dir and use the plain default again.
  # --reset-scale is ignored because --deterministic-reset is set (episode
  # starts from nominal default parameters instead)
  # --surrogate/--dataset are surrogate-only (see test_policy_surrogateEnv.sh);
  # unused with --env tracewin
