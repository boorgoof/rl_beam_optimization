#!/usr/bin/env bash
# Test all 30 SAC grid policies on TraceWin workspace 5: one 30-step episode,
# saving only the final render (including the real TraceWin phase space).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source beam_optimization/.venv/bin/activate

GRID_ROOT="beam_optimization/results/train/rl/sac_penalty_horizon_grid_new"
TEST_ROOT="beam_optimization/results/test/sac_penalty_horizon_grid_tracewin_workspace5_1episode_30steps_final_render"
SURROGATE="beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_018_0.pt"
DATASET="beam_optimization/env/dataset/018/dataset_all.pt"
TRACEWIN_PROJECT="beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_5/CB_newMRMS_RFQ_Fields_1.ini"
TEST_SEED=2026

mapfile -t POLICIES < <(
  find "$GRID_ROOT" -mindepth 3 -maxdepth 3 \
    -type f -path '*/sac/sac_agent.zip' | sort
)

if [ "${#POLICIES[@]}" -ne 30 ]; then
  echo "Expected 30 completed SAC grid checkpoints; found ${#POLICIES[@]}."
  exit 1
fi

for POLICY in "${POLICIES[@]}"; do
  LABEL="$(basename "$(dirname "$(dirname "$POLICY")")")"
  OUTPUT="$TEST_ROOT/$LABEL/test.json"
  CALC_DIR="$TEST_ROOT/$LABEL/tracewin_calc"

  if [ -f "$OUTPUT" ]; then
    echo "Test already complete, skipping: $LABEL"
    continue
  fi

  python -m beam_optimization test \
    --algo sac \
    --policy "$POLICY" \
    --env tracewin \
    --surrogate "$SURROGATE" \
    --dataset "$DATASET" \
    --tracewin-project "$TRACEWIN_PROJECT" \
    --calc-dir "$CALC_DIR" \
    --tracewin-timeout 180 \
    --no-kill-stale \
    --max-ep-steps 30 \
    --episodes 1 \
    --seed "$TEST_SEED" \
    --reset-scale test \
    --output "$OUTPUT" \
    --render \
    --render-final-only \
    --render-dir "$TEST_ROOT/$LABEL/render" \
    --no-episode-video
done

python -m beam_optimization.scripts.summarize_test_grid \
  --test-root "$TEST_ROOT"
