#!/bin/bash
# Train and benchmark:
#   1. SAC + SB3-SAC for three reward-regularization configurations at
#      horizons 20 and 40;
#   2. MBPO + both SVG variants without reward regularization at horizon 20.
#
# Every training run uses dataset/surrogate 018, 200k transitions per agent
# and seed 42. All training finishes before the TraceWin benchmarks begin.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

SURROGATE=beam_optimization/env/surrogate_env/surrogate/trained_models/base/surrogate_018_0.pt
DATASET=beam_optimization/env/dataset/018/dataset_train.pt
TRACEWIN_PROJECT=beam_optimization/env/tracewin_env/tracewin/TraceWin_workspace_2/CB_newMRMS_RFQ_Fields_1.ini
TRAIN_ROOT=beam_optimization/results/train/rl/penalty_matrix_018
BENCHMARK_ROOT=beam_optimization/results/benchmark/penalty_matrix_018
ENSEMBLE_DIR="$TRAIN_ROOT/ensemble_018"

mkdir -p "$ENSEMBLE_DIR"
if [ ! -e "$ENSEMBLE_DIR/surrogate_018_0.pt" ]; then
  ln -s "$(realpath "$SURROGATE")" "$ENSEMBLE_DIR/surrogate_018_0.pt"
fi

benchmark_sac_pair() {
  local run_name="$1"
  local horizon="$2"
  local train_output="$TRAIN_ROOT/$run_name"
  local benchmark_dir="$BENCHMARK_ROOT/$run_name"

  mkdir -p "$benchmark_dir"
  python -m beam_optimization benchmark \
    --policy-only \
    --tracewin-only \
    --surrogate "$SURROGATE" \
    --dataset "$DATASET" \
    --sac "$train_output/sac/sac_agent.pt" \
    --sb3-sac "$train_output/sb3_sac/sb3_sac_agent.zip" \
    --tracewin "$TRACEWIN_PROJECT" \
    --tracewin-episodes 5 \
    --max-ep-steps "$horizon" \
    --policy-seed 42 \
    --hidden 256 256 \
    --output "$benchmark_dir/benchmark_tracewin.json"
}

train_sac_pair() {
  local run_name="$1"
  local horizon="$2"
  local distance_weight="$3"
  local action_weight="$4"
  local regression_weight="$5"
  local train_output="$TRAIN_ROOT/$run_name"

  python -m beam_optimization train_policies \
    --single-surrogate "$SURROGATE" \
    --dataset "$DATASET" \
    --output "$train_output" \
    --rl-steps 200000 \
    --max-ep-steps "$horizon" \
    --hidden 256 256 \
    --seed 42 \
    --n-seeds 1 \
    --eval-every 1000 \
    --eval-episodes 5 \
    --distance-penalty-weight "$distance_weight" \
    --action-penalty-weight "$action_weight" \
    --score-regression-penalty-weight "$regression_weight" \
    --skip td3 ppo ddpg a2c reinforce trpo dyna svg

}

# Horizon 20:
#   all       = KNN + action damping + score-regression damping
#   damping   = action damping + score-regression damping, no KNN
#   knn       = KNN only, no damping
train_sac_pair h20_all 20 0.02 0.02 1.0
train_sac_pair h20_damping 20 0.0 0.02 1.0
train_sac_pair h20_knn 20 0.02 0.0 0.0

# Repeat the same three configurations with 40-step episodes.
train_sac_pair h40_all 40 0.02 0.02 1.0
train_sac_pair h40_damping 40 0.0 0.02 1.0
train_sac_pair h40_knn 40 0.02 0.0 0.0

# MBPO and SVG are intentionally trained without any reward regularization.
MODEL_BASED_NAME=model_based_h20_no_penalties
MODEL_BASED_OUTPUT="$TRAIN_ROOT/$MODEL_BASED_NAME"
MODEL_BASED_BENCHMARK_DIR="$BENCHMARK_ROOT/$MODEL_BASED_NAME"

python -m beam_optimization train_policies \
  --single-surrogate "$SURROGATE" \
  --base-ensemble "$ENSEMBLE_DIR" \
  --dataset "$DATASET" \
  --output "$MODEL_BASED_OUTPUT" \
  --rl-steps 200000 \
  --svg-episodes 10000 \
  --svg-horizon 20 \
  --rollout-length 1 \
  --max-ep-steps 20 \
  --hidden 256 256 \
  --seed 42 \
  --n-seeds 1 \
  --eval-every 1000 \
  --eval-episodes 5 \
  --distance-penalty-weight 0.0 \
  --action-penalty-weight 0.0 \
  --score-regression-penalty-weight 0.0 \
  --skip sac td3 ppo ddpg a2c reinforce trpo sb3_sac

# All training is now complete. Run the seven isolated TraceWin benchmarks.
benchmark_sac_pair h20_all 20
benchmark_sac_pair h20_damping 20
benchmark_sac_pair h20_knn 20
benchmark_sac_pair h40_all 40
benchmark_sac_pair h40_damping 40
benchmark_sac_pair h40_knn 40

mkdir -p "$MODEL_BASED_BENCHMARK_DIR"
python -m beam_optimization benchmark \
  --policy-only \
  --tracewin-only \
  --surrogate "$SURROGATE" \
  --dataset "$DATASET" \
  --mbpo "$MODEL_BASED_OUTPUT/dyna/dyna_agent.pt" \
  --svg-finale "$MODEL_BASED_OUTPUT/svg_finale/svg_agent.pt" \
  --svg-uniform "$MODEL_BASED_OUTPUT/svg_uniform/svg_agent.pt" \
  --tracewin "$TRACEWIN_PROJECT" \
  --tracewin-episodes 5 \
  --max-ep-steps 20 \
  --policy-seed 42 \
  --hidden 256 256 \
  --output "$MODEL_BASED_BENCHMARK_DIR/benchmark_tracewin.json"
