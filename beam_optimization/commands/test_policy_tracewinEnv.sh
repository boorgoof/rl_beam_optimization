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
  --policy beam_optimization/runs/sac_001/sac/sac_agent.pt \
  --env tracewin \
  --calc-dir beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace/tracewin_calc_test \
  --max-ep-steps 5 \
  --seed 42 \
  --deterministic-reset \
  --render \
  --episode-video \
  --tracewin-phase-space
  # --calc-dir avoids TraceWin_workspace/calc, currently owned by comunian
  # (0755) from a stale run: almalinux can't chmod/clean it. Fix that
  # directory (sudo -u comunian rm -rf .../calc/*) if you want to drop
  # --calc-dir and use the plain default again.
