#!/bin/bash
# Evaluate one trained policy step by step and optionally render figures.
# See: beam_optimization/scripts/evaluate.py, README.md section 4 ("evaluate").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization evaluate \
  --algo sac \
  --policy beam_optimization/runs/all/sac/sac_agent.pt \
  --env surrogate \
  --surrogate beam_optimization/env/surrogate_env/surrogate/trained_models/base \
  --dataset beam_optimization/env/dataset/base/dataset_base.pt \
  --episodes 1 \
  --render
