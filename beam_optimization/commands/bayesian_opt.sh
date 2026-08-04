#!/usr/bin/env bash
# Run Bayesian Optimization directly against TraceWin. The Gaussian Process
# is warm-started from the latest dataset and every new point is evaluated by
# the real TraceWin simulator. A seed sequence is optional.
# See: beam_optimization/scripts/bayesian_opt.py, README.md section 4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

source beam_optimization/.venv/bin/activate

python -m beam_optimization bayesian_opt \
  --dataset beam_optimization/env/dataset/018/dataset_all.pt \
  --n-calls 100 \
  --n-runs 1 \
  --warm-best 10 \
  --warm-diverse 30 \
  --bounds-scale 0.6 \
  --seed 42 \
  --tracewin-particles 10000 \
  --timeout 180.0 \
  --retries 2 \
  --output beam_optimization/results/bayesian_opt/bayesian_opt_018.json \
  --new-samples-output beam_optimization/results/bayesian_opt/bayesian_opt_018_tracewin_samples.pt \
  --merged-dataset-output beam_optimization/results/bayesian_opt/dataset_018_with_bayesian.pt \
  "$@"
  # --dataset is pinned above so a newly created numbered dataset cannot
  # silently change the warm start.
  # --workspace/--tracewin default to DEFAULT_TRACEWIN_INI (mutually exclusive)
  # --calc-dir defaults to a dedicated calc folder inside the resolved workspace
  # --bounds-scale mirrors BAYESIAN_SCALE in adige.py; keep them in sync if you recalibrate
  # --tracewin-seed-base defaults to unset (no random_seed passed to TraceWin)
  # --tracewin-threads defaults to unset (TraceWin's own default thread count)
