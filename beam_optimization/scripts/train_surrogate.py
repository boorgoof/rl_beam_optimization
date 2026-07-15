"""Train new base surrogate checkpoints from an existing dataset
(train-only half of the old setup.py)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# if you want to modify the default paths, you can do so in beam_optimization/config/paths.py
from beam_optimization.config.paths import (
    DEFAULT_BASE_SURROGATE_DIR,
    DEFAULT_SURROGATE_LOG_DIR,
    default_dataset_path,
    latest_numbered_dataset_dir,
)
from beam_optimization.env.surrogate_env.surrogate.model.trainer import train_surrogate


def _default_val_dataset() -> str | None:
    """Default --val-dataset to the latest numbered dataset's val split, if
    one exists. Unlike --train-dataset, this must NOT fall back to the flat
    "all"/base dataset: reusing the training data as validation data would
    silently produce meaningless (leaked) validation metrics. No numbered
    dataset yet means "no validation split available" (None), same as
    before this default existed."""
    latest = latest_numbered_dataset_dir()
    if latest is None:
        return None
    candidate = latest / "dataset_val.pt"
    return str(candidate) if candidate.exists() else None


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train new base surrogate checkpoints from an existing train/val dataset."
    )

    parser.add_argument(
        "--train-dataset",
        default=str(default_dataset_path(prefix="train")),
        metavar="PATH",
        help="Training split .pt dataset (e.g. produced by build_dataset).",
    )
    parser.add_argument(
        "--val-dataset",
        default=_default_val_dataset(),
        metavar="PATH",
        help="Validation split .pt dataset, if available.",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=1,
        help="Number of new surrogate checkpoints to add to --model-dir.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_BASE_SURROGATE_DIR),
        metavar="PATH",
        help="Directory where new surrogate_*.pt checkpoints are appended.",
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
    args = parser.parse_args()

    train_dataset_path = Path(args.train_dataset)
    log_dir = Path(args.log_dir) / train_dataset_path.resolve().parent.name

    trainer_summary = train_surrogate(
        train_dataset_path=train_dataset_path,
        val_dataset_path=args.val_dataset,
        output_dir=args.model_dir,
        n_models=args.n_surrogates,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        overwrite=False,
        log_dir=log_dir,
        enable_tensorboard=not args.no_tensorboard,
    )

    print("\nTRAIN SURROGATE COMPLETE")
    print("Created surrogate checkpoints:")
    for checkpoint in trainer_summary["checkpoints"]:
        print(f"  {checkpoint['path']}")

    print("\nJSON summary:")
    print(json.dumps(trainer_summary, indent=2))


if __name__ == "__main__":
    main()
