"""Sensitivity command — thin CLI entry point for config.utility.sensitivity.

The actual sensitivity computation (TraceWin finite differences, CRN,
stability check, copy-paste report for adige.py) lives entirely in
beam_optimization/config/utility/sensitivity.py and is unchanged here.
"""
from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("beam_optimization.config.utility.sensitivity", run_name="__main__")


if __name__ == "__main__":
    main()
