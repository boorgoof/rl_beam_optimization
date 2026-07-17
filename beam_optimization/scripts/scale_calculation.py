"""Dataset, reset and action scale calculation command."""
from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module(
        "beam_optimization.config.offline_utility.scale_calculation",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
