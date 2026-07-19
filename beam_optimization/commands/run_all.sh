#!/bin/bash
# Canonical end-to-end pipeline for the thesis experiments:
#   1. check              — project health check (fast)
#   2. train_policies.sh  — all algorithms × 3 seeds on the surrogate (several hours)
#   3. benchmark_policies_surrogateEnv.sh — BO/SVG + custom-SAC policy benchmark (~1 hour)
#   4. test_policy_surrogateEnv.sh        — one qualitative rendered episode with the trained SAC
#
# Prerequisites (run manually, in order, editing adige.py between steps 1-2):
#   optional/sensitivity.sh -> (paste sensitivity= values into adige.py)
#   bayesian_opt.sh -> (paste default= values into adige.py)
#   build_dataset.sh
#   train_surrogate.sh
#
# For the optional real-physics validation, run benchmark_policies_tracewinEnv.sh
# and test_policy_tracewinEnv.sh afterwards.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization check
bash "$SCRIPT_DIR/train_policies.sh"
bash "$SCRIPT_DIR/benchmark_policies_surrogateEnv.sh"
bash "$SCRIPT_DIR/test_policy_surrogateEnv.sh"
