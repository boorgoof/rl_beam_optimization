#!/usr/bin/env bash
# Evaluate every surrogate_*.pt checkpoint on the test split, including
# per-stage/per-feature errors, final-score accuracy, correlations and plots.
# See: beam_optimization/env/surrogate_env/surrogate/model/evaluator.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

source beam_optimization/.venv/bin/activate

# <-- MODIFY HERE to evaluate against a different dataset's test split.
DATASET_DIR="beam_optimization/env/dataset/018"
MODEL_DIR="beam_optimization/env/surrogate_env/surrogate/trained_models/base_018_final"

python -m beam_optimization evaluate_surrogate \
  --model-dir "$MODEL_DIR" \
  --dataset "$DATASET_DIR/dataset_test.pt" \
  --batch-size 1024 \
  --output beam_optimization/results/benchmark/surrogate_eval_018.json \
  "$@"
  # --device defaults to unset (evaluator auto-selects cuda if available, else cpu)
  # --plots-dir defaults to <output_stem>_plots next to --output
