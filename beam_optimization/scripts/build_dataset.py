"""Create a new TraceWin dataset (build-only half of the old setup.py)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# if you want to modify the default paths, you can do so in beam_optimization/config/paths.py
from beam_optimization.config.paths import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_TRACEWIN_INI,
    TRACEWIN_PROJECT_FILENAME,
    default_tracewin_calc_dir,
    resolve_tracewin_project,
)
from beam_optimization.env.dataset import TraceWinDatasetBuilder, next_numbered_dataset_dir
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import TraceWinSimulator


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Create a fresh numbered TraceWin dataset (train/val/test/all splits)."
    )

    parser.add_argument(
        "--target-samples",
        type=int,
        required=True,
        help="Number of valid TraceWin samples to collect in the new dataset.",
    )
    tracewin_source = parser.add_mutually_exclusive_group()
    tracewin_source.add_argument(
        "--workspace",
        default=None,
        metavar="PATH",
        help=(
            "TraceWin workspace containing "
            f"{TRACEWIN_PROJECT_FILENAME}. Mutually exclusive with --tracewin."
        ),
    )
    tracewin_source.add_argument(
        "--tracewin",
        default=None,
        metavar="INI",
        help=(
            "TraceWin project .ini used to generate the dataset. Mutually "
            f"exclusive with --workspace. Default: {DEFAULT_TRACEWIN_INI}"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        metavar="PATH",
        help="Root where the next numbered dataset directory is created.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        metavar="PATH",
        help=(
            "Exact dataset directory to write to. Pass an existing numbered "
            "directory (e.g. env/dataset/004) to resume an interrupted build "
            "from its builder_state.json. If omitted, a fresh numbered "
            "directory is created under --dataset-root."
        ),
    )
    parser.add_argument(
        "--calc-dir",
        default=None,
        metavar="PATH",
        help=(
            "TraceWin calculation directory. Default: tracewin_calc_<dataset> "
            "inside the TraceWin workspace."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Parameter-sampling seed. Omitted: generate a random seed and save "
            "it in builder_state.json; provided: make sampling reproducible."
        ),
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--no-kill-stale",
        action="store_true",
        help="Do not kill stale TraceWin processes before each simulation.",
    )
    args = parser.parse_args()

    try:
        tracewin_workspace, tracewin_project = resolve_tracewin_project(
            workspace=args.workspace,
            tracewin=args.tracewin,
        )
    except ValueError as exc:
        parser.error(str(exc))

    dataset_root = Path(args.dataset_root)
    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir
        else next_numbered_dataset_dir(dataset_root)
    )
    calc_dir = (
        Path(args.calc_dir)
        if args.calc_dir
        else default_tracewin_calc_dir(tracewin_workspace, dataset_dir)
    )

    print(f"TraceWin workspace: {tracewin_workspace}")
    print(f"TraceWin project:   {tracewin_project}")

    simulator = TraceWinSimulator(
        project_file=str(tracewin_project),
        calc_dir=str(calc_dir),
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        kill_stale=not args.no_kill_stale,
    )

    builder = TraceWinDatasetBuilder(
        simulator,
        output_dir=dataset_dir,
        target_samples=args.target_samples,
        seed=args.seed,
        save_all=True,
        prefix="dataset",
    )
    print(f"Dataset sampling seed: {builder.seed}")
    dataset_summary = builder.build()

    print("\nBUILD DATASET COMPLETE")
    print(f"Created dataset: {dataset_summary['output_dir']}")
    for name, path in sorted(dataset_summary["paths"].items()):
        print(f"  {name}: {path}")

    print("\nJSON summary:")
    print(json.dumps(dataset_summary, indent=2))


if __name__ == "__main__":
    main()
