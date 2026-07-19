#!/bin/bash
# Run the surrogate benchmark and additionally validate the trained custom SAC
# on the REAL TraceWin environment in TraceWin_workspace_2.
# Approximate upper bound for TraceWin alone:
# 1 policy × 10 episodes × 20 steps × ~30 s = ~100 minutes.
# Requires surrogate_004_0.pt, a completed SAC training run, and TraceWin setup.
# See: beam_optimization/scripts/benchmark.py, README.md section 4 ("benchmark").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization benchmark \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_004_0.pt \
  --dataset beam_optimization/env/dataset/004/dataset_all.pt \
  --sac beam_optimization/runs/all/sac/sac_agent.pt \
  --n-runs 3 \
  --eval-budget 200 \
  --svg-episodes 500 \
  --policy-episodes 50 \
  --max-ep-steps 20 \
  --tracewin beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_2/CB_newMRMS_RFQ_Fields_1.ini \
  --tracewin-episodes 10 \
  --output beam_optimization/results/benchmark_tracewin.json
