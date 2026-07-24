#!/bin/bash
# Test one trained policy on one episode and optionally render figures.
# See: beam_optimization/scripts/test.py, README.md section 4 ("test").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization test \
  --algo sac \
  --policy beam_optimization/results/train/rl/sac_001/sac/sac_agent.pt \
  --env surrogate \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base \
  --max-ep-steps 20 \
  --hidden 256 256 \
  --seed 42 \
  --reset-scale test \
  --output beam_optimization/results/test_RL/test.json \
  --render \
  --render-dir beam_optimization/results/test_RL/renders \
  --render-every 1 \
  --dpi 130 \
  --episode-video \
  --episode-video-fps 2
  # --dataset defaults to the latest numbered dataset in env/dataset/ (or
  # the next one to be built, if none exist yet); pass --dataset <path> to pin one
  # --deterministic-reset is off by default (randomized Gaussian reset, per --reset-scale)
  # --tracewin-project/--calc-dir/--tracewin-timeout/--tracewin-phase-space are
  # tracewin-only (see test_policy_tracewinEnv.sh); unused with --env surrogate
  # --max-particles/--bins (40000/150) only affect TraceWin phase-space plots
