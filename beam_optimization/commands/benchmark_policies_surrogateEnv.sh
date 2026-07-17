#!/bin/bash
# Full benchmark on the surrogate: Bayesian optimization + SVG training
# runs, then the final policy benchmark over all trained checkpoints
# (50 independent episodes each). Requires a completed train_policies.sh run.
# See: beam_optimization/scripts/benchmark.py, README.md section 4 ("benchmark").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization benchmark \
  --output beam_optimization/results/benchmark.json \
  --n-runs 3 \
  --eval-budget 3000 \
  --svg-episodes 500 \
  --policy-episodes 50 \
  --sac beam_optimization/runs/all/sac/sac_agent.pt \
  --td3 beam_optimization/runs/all/td3/td3_agent.pt \
  --ppo beam_optimization/runs/all/ppo/ppo_agent.pt \
  --ddpg beam_optimization/runs/all/ddpg/ddpg_agent.pt \
  --a2c beam_optimization/runs/all/a2c/a2c_agent.pt \
  --reinforce beam_optimization/runs/all/reinforce/reinforce_agent.pt \
  --trpo beam_optimization/runs/all/trpo/trpo_agent.pt \
  --sb3-sac beam_optimization/runs/all/sb3_sac/sb3_sac_agent.zip \
  --mbpo beam_optimization/runs/all/dyna/dyna_agent.pt \
  --svg-finale beam_optimization/runs/all/svg_finale/svg_agent.pt \
  --svg-uniform beam_optimization/runs/all/svg_uniform/svg_agent.pt
  # --surrogate defaults to the first surrogate_*.pt found in
  # trained_models/base/; pass --surrogate <path> to pin one
  # --dataset defaults to the latest numbered dataset in env/dataset/ (or
  # the next one to be built, if none exist yet); pass --dataset <path> to pin one
