#!/bin/bash
# Run Bayesian Optimization directly against TraceWin. The Gaussian Process
# is warm-started from the latest dataset and every new point is evaluated by
# the real TraceWin simulator. A seed sequence is optional.
# See: beam_optimization/scripts/bayesian_opt.py, README.md section 4.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization bayesian_opt \
  --n-calls 100 \
  --n-runs 1 \
  --warm-best 10 \
  --warm-diverse 30 \
  --bounds-scale 0.35 \
  --seed 42 \
  --tracewin-particles 10000 \
  --timeout 180.0 \
  --retries 2 \
  --output beam_optimization/results/bayesian_opt/bayesian_opt.json \
  --new-samples-output beam_optimization/results/bayesian_opt/bayesian_opt_tracewin_samples.pt \
  --merged-dataset-output beam_optimization/results/bayesian_opt/dataset_with_bayesian.pt \
  "$@"
  # --dataset defaults to the latest numbered dataset in env/dataset/ (or
  # the next one to be built, if none exist yet); pass --dataset <path> to pin one
  # --workspace/--tracewin default to DEFAULT_TRACEWIN_INI (mutually exclusive)
  # --calc-dir defaults to a dedicated calc folder inside the resolved workspace
  # --bounds-scale mirrors BAYESIAN_SCALE in adige.py; keep them in sync if you recalibrate
  # --tracewin-seed-base defaults to unset (no random_seed passed to TraceWin)
  # --tracewin-threads defaults to unset (TraceWin's own default thread count)
