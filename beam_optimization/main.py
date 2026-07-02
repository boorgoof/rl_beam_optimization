"""Command line entrypoint for beam_optimization."""
from __future__ import annotations

import argparse
import runpy
import sys
from typing import Sequence


COMMAND_MODULES = {
    "check": "beam_optimization.scripts.check",
    "evaluate": "beam_optimization.scripts.evaluate",
    "setup": "beam_optimization.scripts.setup",
    "train": "beam_optimization.scripts.train",
    "benchmark": "beam_optimization.scripts.benchmark",
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog="python -m beam_optimization",
        description="Beam optimization command line tools.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(COMMAND_MODULES),
        help="Command to run.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the selected command.",
    )

    if not argv:
        parser.print_help()
        return 0

    ns = parser.parse_args(argv[:1])
    if ns.command is None:
        parser.print_help()
        return 0

    command_args = argv[1:]
    sys.argv = [f"python -m beam_optimization {ns.command}", *command_args]
    runpy.run_module(COMMAND_MODULES[ns.command], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
