"""CLI entry point for the complete offline surrogate evaluator.

Per-stage/per-feature errors, final-score metrics and plots live in
``env/surrogate_env/surrogate/model/evaluator.py``.
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
