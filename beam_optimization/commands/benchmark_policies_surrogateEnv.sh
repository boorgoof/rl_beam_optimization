#!/bin/bash
# Full benchmark on the surrogate: Bayesian optimization + SVG training,
# followed by 50 deterministic-evaluation episodes for the trained custom SAC.
# Requires surrogate_013_1.pt and a completed SAC training run.
# See: beam_optimization/scripts/benchmark.py, README.md section 4 ("benchmark").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization benchmark \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_013_1.pt \
  --dataset beam_optimization/env/dataset/013/dataset_all.pt \
  --sac beam_optimization/results/train/rl/all/sac/sac_agent.pt \
  --n-runs 3 \
  --eval-budget 200 \
  --svg-baseline \
  --svg-episodes 500 \
  --svg-horizon 20 \
  --policy-episodes 50 \
  --max-ep-steps 20 \
  --policy-seed 42 \
  --tracewin-episodes 5 \
  --hidden 256 256 \
  --output beam_optimization/results/benchmark/benchmark_surrogate.json
  # --tracewin omitted: no real-TraceWin validation in this variant (see
  # benchmark_policies_tracewinEnv.sh); --tracewin-episodes is unused without it.
  # --no-policy-plots/--quick are off by default (full plots, full budget).
  # Other optional trained-checkpoint flags (all unset/not benchmarked unless
  # passed): --td3 --ppo --ddpg --a2c --reinforce --trpo --sb3-sac --mbpo
  # --svg-finale --svg-uniform.
  # --classifier-path defaults to unset (no gating of the Bayesian-Optimization
  # baseline's score() calls near the all-particles-lost cliff); pass
  # --classifier-path <failure_classifier_*.pt> to enable it, matching the
  # dataset --surrogate above was trained on.
  # --svg-baseline trains a fresh SVGAgent from scratch (both stage-weight
  # variants) as a comparison method, in addition to Bayesian Optimization --
  # off by default in benchmark.py, explicitly enabled here to keep this
  # script's original full BO+SVG+policy comparison.
