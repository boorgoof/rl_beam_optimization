#!/bin/bash
# Validate the best trained policies on the REAL TraceWin environment
# (~30 s per step: 3 policies × 5 episodes × 20 steps ≈ 2.5 hours).
# Requires the local TraceWin setup (README.md section 2) and a completed
# train_policies.sh run. Pass only the checkpoints you want to validate.
# See: beam_optimization/scripts/benchmark.py, README.md section 4 ("benchmark").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization benchmark \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_0.pt \
  --output beam_optimization/results/benchmark_tracewin.json \
  --quick \
  --tracewin \
  --tracewin-episodes 5 \
  --sac beam_optimization/runs/all/sac/sac_agent.pt \
  --mbpo beam_optimization/runs/all/dyna/dyna_agent.pt \
  --svg-finale beam_optimization/runs/all/svg_finale/svg_agent.pt
  # --dataset defaults to the latest numbered dataset in env/dataset/ (falls
  # back to env/dataset/base/dataset_base.pt); pass --dataset <path> to pin one
