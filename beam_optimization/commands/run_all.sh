#!/bin/bash
# Canonical end-to-end pipeline for the thesis experiments:
#   1. check      — project health check (fast)
#   2. train.sh   — all algorithms × 3 seeds on the surrogate (several hours)
#   3. benchmark.sh — BO/SVG + policy benchmark on all checkpoints (~1 hour)
#   4. test.sh    — one qualitative rendered episode with the trained SAC
# Prerequisite: base dataset and surrogates exist (otherwise run setup.sh first).
# For the optional real-physics validation, run benchmark_tracewin.sh afterwards.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization check
bash "$SCRIPT_DIR/train.sh"
bash "$SCRIPT_DIR/benchmark.sh"
bash "$SCRIPT_DIR/test.sh"
