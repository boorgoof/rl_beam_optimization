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
  --policy beam_optimization/runs/all/sac/sac_agent.pt \
  --env tracewin \
  --max-ep-steps 5 \
  --seed 42 \
  --deterministic-reset \
  --render \
  --episode-video \
  --tracewin-phase-space
