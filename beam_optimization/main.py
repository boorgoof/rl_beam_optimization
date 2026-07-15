"""Command line entrypoint for beam_optimization."""
from __future__ import annotations

import argparse
import runpy
import sys
from typing import Sequence

# associate commands with their module names so that we can run them with runpy
COMMAND_MODULES = {
    "check": "beam_optimization.scripts.check",
    "sensitivity": "beam_optimization.scripts.sensitivity",
    "sensitivity2": "beam_optimization.scripts.sensitivity2",
    "sensitivity3": "beam_optimization.scripts.sensitivity3",
    "build_dataset": "beam_optimization.scripts.build_dataset",
    "train_surrogate": "beam_optimization.scripts.train_surrogate",
    "evaluate_surrogate": "beam_optimization.scripts.evaluate_surrogate",
    "bayesian_opt": "beam_optimization.scripts.bayesian_opt",
    "train_policies": "beam_optimization.scripts.train_policies",
    "benchmark": "beam_optimization.scripts.benchmark",
    "test": "beam_optimization.scripts.test",
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    
    # create the parser 
    parser = argparse.ArgumentParser(
        prog="python -m beam_optimization",
        description="Beam optimization command line tools.",
    )
    # add the command argument and the remaining arguments
    # for example if the user runs `python -m beam_optimization train_policies --rl-steps 100`,
    # then `train_policies` is the command
    # and `--rl-steps 100` are the remaining arguments
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
