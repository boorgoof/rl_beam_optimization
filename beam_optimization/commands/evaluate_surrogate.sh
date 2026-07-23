#!/bin/bash
# Evaluate every surrogate_*.pt checkpoint on the test split, including
# per-stage/per-feature errors, final-score accuracy, correlations and plots.
# See: beam_optimization/env/surrogate_env/surrogate/model/evaluator.py.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization evaluate_surrogate \
  --model-dir beam_optimization/env/surrogate_env/surrogate/trained_models/base \
  --output beam_optimization/results/benchmark/surrogate_eval.json \
  "$@"
  # --dataset defaults to the latest numbered dataset's test split; pass
  # --dataset <path> to pin a specific numbered dataset.
