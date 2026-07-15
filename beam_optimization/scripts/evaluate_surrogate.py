"""Evaluate surrogate command — thin CLI entry point for the evaluator module.

The actual evaluation (per-stage MSE/RMSE over surrogate_*.pt checkpoints)
lives entirely in
beam_optimization/env/surrogate_env/surrogate/model/evaluator.py and is
unchanged here.
"""
from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module(
        "beam_optimization.env.surrogate_env.surrogate.model.evaluator",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
