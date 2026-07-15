#!/bin/bash
# Quick single-seed sensitivity estimate (no averaging over repeats) — much
# faster than sensitivity.sh, at the cost of not being able to assess how
# stable the estimate is. Use sensitivity.sh instead when time allows.
# See: beam_optimization/config/utility/sensitivity2.py, README.md section 4.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ -f "beam_optimization/.venv/bin/activate" ]; then
  source beam_optimization/.venv/bin/activate
fi

python -m beam_optimization sensitivity2
