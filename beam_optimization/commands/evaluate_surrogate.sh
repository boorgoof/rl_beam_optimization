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
  --classifier-path "$MODEL_DIR/failure_classifier_$(basename "$DATASET_DIR").pt" \
  --batch-size 1024 \
  --output beam_optimization/results/benchmark/surrogate_eval_018.json \
  "$@"
  # --classifier-path is set by default (train_surrogate trains one shared
  # classifier per dataset automatically) so classifier_metrics,
  # score_metrics_gated, and the classifier_proba_hist/classifier_confusion
  # plots are always produced; if failure_classifier_<dataset>.pt doesn't
  # exist yet (e.g. trained with --skip-classifier), either train one or
  # comment out the --classifier-path line above -- evaluate_surrogate.py
  # requires the file to exist when the flag is given, it does not fall
  # back to "no classifier" on a missing path.
  # --device defaults to unset (evaluator auto-selects cuda if available, else cpu)
  # --plots-dir defaults to <output_stem>_plots next to --output
