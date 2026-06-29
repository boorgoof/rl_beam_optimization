"""Build new flat .pt datasets by running TraceWin simulations.

This module is for offline dataset generation. It creates fresh BeamDataset
objects in memory and writes new .pt files; it never appends to an existing
dataset file. If no output directory is provided, datasets are written under
env/dataset/001, env/dataset/002, ... while env/dataset/base remains the base
dataset used for beam0 sampling and online updates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from beam_optimization.config.adige import PARAM_KEYS, action_bounds, default_params
from beam_optimization.config.paths import DEFAULT_DATASET_ROOT
from beam_optimization.env.dataset.dataset import BeamDataset
from beam_optimization.env.dataset.utility import tracewin_result_to_flat_sample
from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator


SplitRatios = tuple[float, float, float]


def next_numbered_dataset_dir(
    root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    width: int = 3,
) -> Path:
    """Return the next numbered dataset directory under root.

    Example:
        if ``env/dataset/001`` and ``env/dataset/002`` exist, return
        ``env/dataset/003``. Non-numeric folders such as ``base`` are ignored.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    used_numbers = []
    for child in root.iterdir():
        if child.is_dir() and child.name.isdigit():
            used_numbers.append(int(child.name))

    next_idx = max(used_numbers, default=0) + 1
    return root / f"{next_idx:0{width}d}"


def sample_parameter_sets(
    n_samples: int,
    *,
    action_scale: float = 1.0,
    seed: Optional[int] = None,
) -> list[Dict[str, float]]:
    """Sample parameter dictionaries around ADIGE defaults.

    The sampling range is default +/- action_scale * sensitivity for each
    parameter, matching the environment action scaling.
    """
    rng = np.random.default_rng(seed)
    low_delta, high_delta = action_bounds(action_scale)
    defaults = default_params()

    param_sets: list[Dict[str, float]] = []
    for _ in range(int(n_samples)):
        deltas = rng.uniform(low_delta, high_delta).astype(np.float32)
        params = {
            key: float(defaults[key] + delta)
            for key, delta in zip(PARAM_KEYS, deltas)
        }
        param_sets.append(params)
    return param_sets


def dataset_from_tracewin_results(
    results: Iterable[BeamSimulationResult],
    *,
    skip_failed: bool = True,
) -> BeamDataset:
    """Convert TraceWin results into one fresh BeamDataset."""
    dataset = BeamDataset()
    for result in results:
        if skip_failed and (
            result.source != "tracewin" or not result.success or result.beam_states is None
        ):
            continue
        x, y, score = tracewin_result_to_flat_sample(result)
        dataset.append_flat_sample(x, y, score)
    return dataset


def split_dataset(
    dataset: BeamDataset,
    *,
    ratios: SplitRatios = (0.8, 0.1, 0.1),
    seed: Optional[int] = 123,
    shuffle: bool = True,
) -> dict[str, BeamDataset]:
    """Split a BeamDataset into train/val/test BeamDataset objects."""
    n_samples = len(dataset)
    train_ratio, val_ratio, test_ratio = ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if n_samples == 0:
        raise ValueError("Cannot split an empty dataset")
    if ratio_sum <= 0:
        raise ValueError(f"Split ratios must sum to a positive value, got {ratios}")

    normalized = np.asarray(ratios, dtype=np.float64) / ratio_sum
    n_train = int(np.floor(n_samples * normalized[0]))
    n_val = int(np.floor(n_samples * normalized[1]))
    n_test = n_samples - n_train - n_val

    if n_samples >= 3:
        if n_train == 0:
            n_train, n_test = 1, max(0, n_test - 1)
        if n_val == 0:
            n_val, n_train = 1, max(0, n_train - 1)
        if n_test == 0:
            n_test, n_train = 1, max(0, n_train - 1)

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    boundaries = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:n_train + n_val + n_test],
    }

    splits: dict[str, BeamDataset] = {}
    for name, idx in boundaries.items():
        split = BeamDataset()
        if len(idx) > 0:
            split.append_flat_samples(
                dataset.X[idx],
                dataset.Y[idx],
                dataset.scores[idx],
            )
        splits[name] = split
    return splits


def save_dataset_splits(
    dataset: BeamDataset,
    output_dir: str | Path,
    *,
    split: bool = True,
    ratios: SplitRatios = (0.8, 0.1, 0.1),
    save_all: bool = False,
    seed: Optional[int] = 123,
    prefix: str = "dataset",
) -> dict[str, Path]:
    """Save a BeamDataset as all/train/val/test .pt files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    if save_all or not split:
        path = output_dir / f"{prefix}_all.pt"
        dataset.save_flat(path)
        saved["all"] = path

    if split:
        for name, split_ds in split_dataset(dataset, ratios=ratios, seed=seed).items():
            path = output_dir / f"{prefix}_{name}.pt"
            split_ds.save_flat(path)
            saved[name] = path

    return saved


def build_tracewin_dataset(
    simulator: BeamSimulator,
    output_dir: Optional[str | Path] = None,
    *,
    param_sets: Optional[Sequence[Dict[str, float]]] = None,
    n_samples: Optional[int] = None,
    action_scale: float = 1.0,
    split: bool = True,
    split_ratios: SplitRatios = (0.8, 0.1, 0.1),
    save_all: bool = False,
    seed: Optional[int] = 123,
    skip_failed: bool = True,
    prefix: str = "dataset",
) -> dict:
    """Run TraceWin and save a fresh dataset.

    Args:
        simulator: TraceWinSimulator or any BeamSimulator with simulate(params).
        output_dir: Destination directory for .pt files. If None, a new
            numbered directory is created under env/dataset, e.g. 001, 002, ...
        param_sets: Optional explicit list of parameter dictionaries.
        n_samples: Number of random parameter sets to generate when param_sets
            is not provided.
        action_scale: Sampling range multiplier for generated param_sets.
        split: If True, save train/val/test files.
        split_ratios: Train/val/test ratios. Defaults to 80/10/10.
        save_all: If True, also save dataset_all.pt.
        seed: Random seed for sampling and splitting.
        skip_failed: If True, failed TraceWin runs are ignored.
        prefix: Filename prefix, e.g. "dataset" -> dataset_train.pt.

    Returns:
        Dictionary with counts and saved file paths.
    """
    if param_sets is None:
        if n_samples is None:
            raise ValueError("Provide either param_sets or n_samples")
        param_sets = sample_parameter_sets(
            n_samples,
            action_scale=action_scale,
            seed=seed,
        )

    if output_dir is None:
        output_dir = next_numbered_dataset_dir()

    results = [simulator.simulate(params) for params in param_sets]
    dataset = dataset_from_tracewin_results(results, skip_failed=skip_failed)
    saved_paths = save_dataset_splits(
        dataset,
        output_dir,
        split=split,
        ratios=split_ratios,
        save_all=save_all,
        seed=seed,
        prefix=prefix,
    )

    return {
        "n_requested": len(param_sets),
        "n_success": len(dataset),
        "n_failed": len(param_sets) - len(dataset),
        "paths": {name: str(path) for name, path in saved_paths.items()},
    }
