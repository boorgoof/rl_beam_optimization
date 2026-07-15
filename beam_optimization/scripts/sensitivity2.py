"""Sensitivity2 command — thin CLI entry point for config.utility.sensitivity2.

Quick single-seed sensitivity estimate; see config/utility/sensitivity2.py for
the actual algorithm (unchanged here).
"""
from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("beam_optimization.config.utility.sensitivity2", run_name="__main__")


if __name__ == "__main__":
    main()
