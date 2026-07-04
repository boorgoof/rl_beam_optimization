"""Command line entrypoint for beam_optimization."""
from __future__ import annotations

import argparse
import runpy
import sys
from typing import Sequence

# associate commands with their module names so that we can run them with runpy
COMMAND_MODULES = {
    "check": "beam_optimization.scripts.check",
    "setup": "beam_optimization.scripts.setup",
    "test": "beam_optimization.scripts.test",
    "train": "beam_optimization.scripts.train",
    "benchmark": "beam_optimization.scripts.benchmark",
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    
    # create the parser 
    parser = argparse.ArgumentParser(
        prog="python -m beam_optimization",
        description="Beam optimization command line tools.",
    )
    # add the command argument and the remaining arguments
    # for example if the user runs `python -m beam_optimization train --episodes 100`, 
    # then `train` is the command 
    # and `--episodes 100` are the remaining arguments
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

    # if no command is provided, print help and exit
    if not argv:
        parser.print_help()
        return 0

    # parse the command and run the corresponding module
    parsed_command = parser.parse_args(argv[:1])
    if parsed_command.command is None:
        parser.print_help()
        return 0

    # run the selected command module with the remaining arguments
    command_args = argv[1:]
    sys.argv = [f"python -m beam_optimization {parsed_command.command}", *command_args]
    runpy.run_module(COMMAND_MODULES[parsed_command.command], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
