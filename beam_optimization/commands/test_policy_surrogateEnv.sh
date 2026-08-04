#!/usr/bin/env bash
# Test one trained policy on one episode and optionally render figures.
# See: beam_optimization/scripts/test.py, README.md section 4 ("test").
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

source beam_optimization/.venv/bin/activate

python -m beam_optimization test \
  --algo sac \
  --policy beam_optimization/results/train/rl/final_thesis_018_a030_s050/sac/sac_agent.zip \
  --env surrogate \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base_018_final/surrogate_018_0.pt \
  --dataset beam_optimization/env/dataset/018/dataset_all.pt \
  --max-ep-steps 20 \
  --hidden 256 256 \
  --seed 42 \
  --reset-scale test \
  --output beam_optimization/results/test_RL/final_thesis_surrogate/test.json \
  --render \
  --render-dir beam_optimization/results/test_RL/final_thesis_surrogate/renders \
  --render-every 1 \
  --dpi 130 \
  --episode-video \
  --episode-video-fps 2 \
  "$@"
  # --deterministic-reset is off by default (randomized Gaussian reset, per --reset-scale)
  # --tracewin-project/--calc-dir/--tracewin-timeout/--tracewin-phase-space are
  # tracewin-only (see test_policy_tracewinEnv.sh); unused with --env surrogate
  # --max-particles/--bins (40000/150) only affect TraceWin phase-space plots
