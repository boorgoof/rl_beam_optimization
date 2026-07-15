"""Create a new TraceWin dataset (build-only half of the old setup.py)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# if you want to modify the default paths, you can do so in beam_optimization/config/paths.py
from beam_optimization.config.paths import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_TRACEWIN_INI,
    default_tracewin_calc_dir,
)
from beam_optimization.env.dataset import TraceWinDatasetBuilder, next_numbered_dataset_dir
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import TraceWinSimulator


TRACEWIN_PROJECT_FILENAME = "CB_newMRMS_RFQ_Fields_1.ini"


def resolve_tracewin_project(
    *,
    workspace: str | None = None,
    tracewin: str | None = None,
) -> tuple[Path, Path]:
    """Return the resolved ``(workspace, project_file)`` selected by the CLI."""
    if workspace is not None and tracewin is not None:
        raise ValueError("--workspace and --tracewin are mutually exclusive")

    if workspace is not None:
        workspace_path = Path(workspace).expanduser().resolve()
        if not workspace_path.is_dir():
            raise ValueError(
                f"TraceWin workspace does not exist or is not a directory: {workspace_path}"
            )
        project_file = workspace_path / TRACEWIN_PROJECT_FILENAME
    else:
        project_file = Path(tracewin or DEFAULT_TRACEWIN_INI).expanduser().resolve()
        workspace_path = project_file.parent

    if not project_file.is_file():
        raise ValueError(f"TraceWin project file not found: {project_file}")

    return workspace_path, project_file


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
        "--calc-dir",
        default=None,
        metavar="PATH",
        help=(
            "TraceWin calculation directory. Default: tracewin_calc inside "
            "the newly created numbered dataset directory."
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
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
    dataset_dir = next_numbered_dataset_dir(dataset_root)
    calc_dir = Path(args.calc_dir) if args.calc_dir else default_tracewin_calc_dir(dataset_dir)

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
    dataset_summary = builder.build()

    print("\nBUILD DATASET COMPLETE")
    print(f"Created dataset: {dataset_summary['output_dir']}")
    for name, path in sorted(dataset_summary["paths"].items()):
        print(f"  {name}: {path}")

    print("\nJSON summary:")
    print(json.dumps(dataset_summary, indent=2))


if __name__ == "__main__":
    main()
