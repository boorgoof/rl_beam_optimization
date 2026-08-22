#!/usr/bin/env bash
# Re-run the final-thesis TraceWin benchmark on workspace 3 with 30-step episodes.
# The seven promoted checkpoints, 20 matched reset seeds, and all other benchmark
# settings are identical to final_thesis_018_a015_s025_steps300k_tracewin.json.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

TRAIN_OUTPUT="beam_optimization/results/train/rl/final_thesis_018_a015_s025_steps300k"
MODEL_DIR="beam_optimization/env/surrogate_env/surrogate/trained_models/base"
TRACEWIN_PROJECT="beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_3/CB_newMRMS_RFQ_Fields_1.ini"
RESULT_DIR="beam_optimization/results/benchmark/final_thesis_018_a015_s025_steps300k_workspace3_30steps"
BENCHMARK_OUTPUT="$RESULT_DIR/final_thesis_018_a015_s025_steps300k_workspace3_30steps_tracewin.json"

mkdir -p "$RESULT_DIR"

if [ -e "$BENCHMARK_OUTPUT" ]; then
  echo "Refusing to overwrite existing benchmark output: $BENCHMARK_OUTPUT" >&2
  exit 1
fi

python -m beam_optimization benchmark \
  --surrogate "$MODEL_DIR/surrogate_018_0.pt" \
  --dataset beam_optimization/env/dataset/018/dataset_all.pt \
  --policy-only \
  --tracewin-only \
  --tracewin "$TRACEWIN_PROJECT" \
  --no-kill-stale \
  --tracewin-threads 6 \
  --tracewin-timeout 180 \
  --sac "$TRAIN_OUTPUT/sac/sac_agent.zip" \
  --ppo "$TRAIN_OUTPUT/ppo/ppo_agent.zip" \
  --td3 "$TRAIN_OUTPUT/td3/td3_agent.zip" \
  --ddpg "$TRAIN_OUTPUT/ddpg/ddpg_agent.zip" \
  --a2c "$TRAIN_OUTPUT/a2c/a2c_agent.zip" \
  --svg-final "$TRAIN_OUTPUT/svg_finale/svg_agent.pt" \
  --svg-uniform "$TRAIN_OUTPUT/svg_uniform/svg_agent.pt" \
  --tracewin-episodes 20 \
  --max-ep-steps 30 \
  --policy-seed 42 \
  --hidden 256 256 \
  --output "$BENCHMARK_OUTPUT"
