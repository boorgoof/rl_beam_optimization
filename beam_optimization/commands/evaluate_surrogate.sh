#!/bin/bash
# Evaluate every surrogate_*.pt checkpoint in a folder against a BeamDataset,
# reporting per-stage MSE/RMSE on beam-state predictions.
# See: beam_optimization/env/surrogate_env/surrogate/model/evaluator.py.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization evaluate_surrogate \
  --model-dir beam_optimization/env/surrogate_env/surrogate/trained_models/base \
  --output beam_optimization/results/surrogate_eval.json
  # --dataset defaults to the latest numbered dataset's val split in
  # env/dataset/ (or the next one to be built, if none exist yet); pass
  # --dataset <path> to pin one
