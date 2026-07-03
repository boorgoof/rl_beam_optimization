"""Create a new TraceWin dataset and add new base surrogate checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# if you want to modify the default paths, you can do so in beam_optimization/config/paths.py
# here are the default paths for the surrogate model checkpoints, dataset root, surrogate training logs (tensorboard/csv), tracewin project .ini and tracewin calculation directory
from beam_optimization.config.paths import (
    DEFAULT_BASE_SURROGATE_DIR,
    DEFAULT_DATASET_ROOT,
    DEFAULT_SURROGATE_LOG_DIR,
    DEFAULT_TRACEWIN_INI,
    default_tracewin_calc_dir,
)
from beam_optimization.env.dataset import TraceWinDatasetBuilder, next_numbered_dataset_dir
from beam_optimization.env.surrogate_env.surrogate.model.trainer import train_surrogate
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import TraceWinSimulator


def main() -> None:
    
    # create the argument parser 
    parser = argparse.ArgumentParser(
        description=(
            "Create a fresh numbered TraceWin dataset and train new base "
            "surrogate checkpoints from it."
        )
    )

    # add the command line arguments 
    parser.add_argument(
        "--target-samples",
        type=int,
        required=True,
        help="Number of valid TraceWin samples to collect in the new dataset.",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=1,
        help="Number of new surrogate checkpoints to add to models/base.",
    )
    parser.add_argument(
        "--tracewin",
        default=str(DEFAULT_TRACEWIN_INI),
        metavar="INI",
        help="TraceWin project .ini used to generate the dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        metavar="PATH",
        help="Root where the next numbered dataset directory is created.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_BASE_SURROGATE_DIR),
        metavar="PATH",
        help="Directory where new surrogate_*.pt checkpoints are appended.",
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
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_SURROGATE_LOG_DIR),
        metavar="PATH",
        help="TensorBoard/CSV log root for surrogate training.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard/metrics.csv logging for surrogate training.",
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

    dataset_root = Path(args.dataset_root)
    dataset_dir = next_numbered_dataset_dir(dataset_root)
    calc_dir = Path(args.calc_dir) if args.calc_dir else default_tracewin_calc_dir(dataset_dir)

    simulator = TraceWinSimulator(
        project_file=args.tracewin,
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

    paths = dataset_summary["paths"]
    train_path = paths["train"]
    val_path = paths.get("val")

    trainer_summary = train_surrogate(
        train_dataset_path=train_path,
        val_dataset_path=val_path,
        output_dir=args.model_dir,
        n_models=args.n_surrogates,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        overwrite=False,
        log_dir=Path(args.log_dir) / dataset_dir.name,
        enable_tensorboard=not args.no_tensorboard,
    )

    print("\nSETUP COMPLETE")
    print(f"Created dataset: {dataset_summary['output_dir']}")
    for name, path in sorted(paths.items()):
        print(f"  {name}: {path}")
    print("Created surrogate checkpoints:")
    for checkpoint in trainer_summary["checkpoints"]:
        print(f"  {checkpoint['path']}")

    print("\nJSON summary:")
    print(
        json.dumps(
            {
                "dataset": dataset_summary,
                "surrogates": trainer_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
