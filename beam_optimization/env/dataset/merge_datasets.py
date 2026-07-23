"""Validate and merge multiple flat BeamDataset files into one dataset."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
from typing import Sequence

import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    N_OUTPUT_STAGES,
    N_PARAMS,
    PARAMETERS,
    STAGE_MARKERS,
    score_from_matrix,
)
from beam_optimization.env.dataset.dataset import BeamDataset
from beam_optimization.env.dataset.tracewin_dataset_builder import save_dataset_splits


_EXPECTED_X_COLS = list(BEAM_STATE_FEATURES) + [parameter.name for parameter in PARAMETERS]
_EXPECTED_Y_COLS = [
    f"{feature}_s{stage}"
    for stage in range(1, N_OUTPUT_STAGES + 1)
    for feature in BEAM_STATE_FEATURES
]
_EXPECTED_MARKERS = list(STAGE_MARKERS)
_X_WIDTH = BEAM_STATE_DIM + N_PARAMS
_Y_WIDTH = N_OUTPUT_STAGES * BEAM_STATE_DIM
_OUTPUT_FILENAMES = (
    "dataset_all.pt",
    "dataset_train.pt",
    "dataset_val.pt",
    "dataset_test.pt",
)


def _resolved(path: Path) -> Path:
    """Resolve a path without requiring it to exist."""
    return path.expanduser().resolve(strict=False)


def _builder_is_running(path: Path) -> bool:
    state_path = path.parent / "builder_state.json"
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read builder state {state_path}: {exc}") from exc
    return str(state.get("status", "")).lower() == "running"


def _load_stable_snapshot(
    path: Path,
    *,
    attempts: int = 20,
    retry_sleep: float = 0.1,
) -> dict:
    """Read one internally consistent snapshot while a builder may replace it."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            unchanged = (
                before.st_ino == after.st_ino
                and before.st_size == after.st_size == len(payload)
                and before.st_mtime_ns == after.st_mtime_ns
            )
            if not unchanged:
                raise RuntimeError("dataset changed while snapshot was being read")
            raw = torch.load(
                io.BytesIO(payload),
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(raw, dict):
                raise RuntimeError("snapshot does not contain a dataset dictionary")
            return raw
        except Exception as exc:
            last_error = exc
            time.sleep(retry_sleep)
    raise ValueError(
        f"Could not obtain a stable snapshot of running dataset {path} "
        f"after {attempts} attempts: {last_error}"
    )


def _as_float_tensor(raw: dict, key: str, path: Path) -> torch.Tensor:
    if key not in raw:
        raise ValueError(f"Invalid dataset {path}: missing required key {key!r}")
    try:
        tensor = torch.as_tensor(raw[key]).detach().cpu().float()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid dataset {path}: {key} is not a numeric tensor") from exc
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Invalid dataset {path}: {key} contains NaN or infinite values")
    return tensor


def _validate_and_load(
    path: Path,
    *,
    allow_running: bool = False,
) -> tuple[BeamDataset, bool]:
    if not path.is_file():
        raise ValueError(f"Input dataset does not exist or is not a file: {path}")
    running = _builder_is_running(path)
    if running and not allow_running:
        state_path = path.parent / "builder_state.json"
        raise ValueError(
            f"Input dataset {path} belongs to a build that is still running "
            f"({state_path}). Finish it before merging, or pass --allow-running "
            "to merge a stable snapshot."
        )

    try:
        raw = (
            _load_stable_snapshot(path)
            if running
            else torch.load(str(path), map_location="cpu", weights_only=False)
        )
    except Exception as exc:
        raise ValueError(f"Cannot load dataset {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid dataset {path}: expected a dictionary stored in the .pt file")

    X = _as_float_tensor(raw, "X", path)
    Y = _as_float_tensor(raw, "Y", path)
    scores = _as_float_tensor(raw, "scores", path).reshape(-1)

    if X.ndim != 2 or tuple(X.shape[1:]) != (_X_WIDTH,):
        raise ValueError(f"Invalid dataset {path}: X must have shape (N, {_X_WIDTH}), got {tuple(X.shape)}")
    if Y.ndim != 2 or tuple(Y.shape[1:]) != (_Y_WIDTH,):
        raise ValueError(f"Invalid dataset {path}: Y must have shape (N, {_Y_WIDTH}), got {tuple(Y.shape)}")
    if X.shape[0] != Y.shape[0] or X.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Invalid dataset {path}: X, Y and scores have different sample counts "
            f"({X.shape[0]}, {Y.shape[0]}, {scores.shape[0]})"
        )

    expected_metadata = {
        "x_cols": _EXPECTED_X_COLS,
        "y_cols": _EXPECTED_Y_COLS,
        "markers": _EXPECTED_MARKERS,
    }
    for key, expected in expected_metadata.items():
        if key not in raw:
            raise ValueError(f"Invalid dataset {path}: missing required metadata {key!r}")
        if list(raw[key]) != expected:
            raise ValueError(
                f"Incompatible dataset {path}: metadata {key!r} does not match "
                "the current BeamDataset schema"
            )

    if "num_samples" in raw and int(raw["num_samples"]) != X.shape[0]:
        raise ValueError(
            f"Invalid dataset {path}: num_samples={raw['num_samples']} but tensors contain {X.shape[0]} rows"
        )

    recalculated_scores = torch.as_tensor(
        score_from_matrix(Y[:, -BEAM_STATE_DIM:].numpy()),
        dtype=torch.float32,
    ).reshape(-1)
    if not torch.isfinite(recalculated_scores).all():
        raise ValueError(f"Invalid dataset {path}: recalculated scores contain NaN or infinite values")

    dataset = BeamDataset()
    dataset.append_flat_samples(X, Y, recalculated_scores)
    return dataset, running


def merge_dataset_files(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    seed: int = 123,
    allow_running: bool = False,
) -> dict:
    """Merge input datasets, then save all/train/val/test files.

    Samples are preserved in input order in ``dataset_all.pt``. The split files
    use the project's standard 80/10/10 shuffled split.
    """
    inputs = [Path(path).expanduser() for path in input_paths]
    if len(inputs) < 2:
        raise ValueError("At least two input dataset files are required")

    output_dir = Path(output_dir).expanduser()
    targets = [output_dir / filename for filename in _OUTPUT_FILENAMES]
    resolved_inputs = [_resolved(path) for path in inputs]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError("The same input dataset was specified more than once")
    resolved_targets = {_resolved(path) for path in targets}
    collisions = [path for path, resolved in zip(inputs, resolved_inputs) if resolved in resolved_targets]
    if collisions:
        raise ValueError(
            "An input dataset cannot also be an output file: "
            + ", ".join(str(path) for path in collisions)
        )
    existing = [path for path in targets if path.exists()]
    if existing:
        raise ValueError(
            "Refusing to overwrite existing merged dataset files: "
            + ", ".join(str(path) for path in existing)
        )

    merged = BeamDataset()
    input_counts: list[dict[str, object]] = []
    for path in inputs:
        dataset, snapshotted = _validate_and_load(
            path,
            allow_running=allow_running,
        )
        merged = merged.merge(dataset)
        input_counts.append({
            "path": str(path),
            "num_samples": len(dataset),
            "running_snapshot": snapshotted,
        })
        suffix = " (stable running snapshot)" if snapshotted else ""
        print(f"Input: {path} -> {len(dataset):,} samples{suffix}", flush=True)

    saved = save_dataset_splits(
        merged,
        output_dir,
        split=True,
        ratios=(0.8, 0.1, 0.1),
        save_all=True,
        seed=int(seed),
        prefix="dataset",
    )

    split_counts = {
        name: len(BeamDataset.load(path))
        for name, path in saved.items()
    }
    print(f"Merged total: {len(merged):,} samples", flush=True)
    print(
        "Splits: "
        + ", ".join(f"{name}={split_counts[name]:,}" for name in ("train", "val", "test")),
        flush=True,
    )
    return {
        "inputs": input_counts,
        "output_dir": str(output_dir),
        "seed": int(seed),
        "allow_running": bool(allow_running),
        "num_samples": len(merged),
        "split_counts": split_counts,
        "paths": {name: str(path) for name, path in saved.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two or more compatible BeamDataset .pt files into one dataset."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="DATASET_PT",
        help="Two or more dataset .pt files, normally dataset_all.pt files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="PATH",
        help="New directory for dataset_all/train/val/test.pt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Shuffle seed used for the 80/10/10 split (default: 123).",
    )
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help=(
            "Allow inputs whose builder_state.json is still running by reading "
            "a stable point-in-time snapshot. Builders continue unchanged."
        ),
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("--inputs requires at least two dataset files")
    try:
        summary = merge_dataset_files(
            args.inputs,
            args.output_dir,
            seed=args.seed,
            allow_running=args.allow_running,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print("\nMERGE DATASETS COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
