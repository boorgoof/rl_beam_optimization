"""Build flat .pt datasets by running TraceWin simulations.

The main entry point is TraceWinDatasetBuilder. It is designed for expensive
offline TraceWin generation: it keeps an incremental dataset_all.pt and a
builder_state.json file so an interrupted run can be resumed until the target
number of valid samples is reached.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from beam_optimization.config.adige import (
    PARAM_KEYS,
    PARAMETERS,
    clip_params_to_hw,
    dataset_std_vec,
    default_params,
)
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
    seed: Optional[int] = None,
) -> list[Dict[str, float]]:
    """Sample parameter dictionaries around ADIGE defaults.

    Each parameter is drawn from N(default_p, dataset_std_p^2), then clipped to
    known hardware bounds. See generate_param_sets_gaussian() for details.
    """
    return generate_param_sets_gaussian(n_samples, oversample_factor=1.0, seed=seed)


def dataset_from_tracewin_results(
    results: Iterable[BeamSimulationResult],
    *,
    skip_failed: bool = True,
) -> BeamDataset:
    """Convert TraceWin results into one fresh BeamDataset."""
    dataset = BeamDataset()
    for result in results:
        if skip_failed and not _is_valid_tracewin_result(result):
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


class TraceWinDatasetBuilder:
    """Resumable offline builder for TraceWin-generated BeamDataset files."""

    STATE_FILENAME = "builder_state.json"

    def __init__(
        self,
        simulator: BeamSimulator,
        output_dir: Optional[str | Path] = None,
        *,
        target_samples: int,
        split_ratios: SplitRatios = (0.8, 0.1, 0.1),
        seed: Optional[int] = 123,
        save_all: bool = True,
        prefix: str = "dataset",
        param_sets: Optional[Sequence[Dict[str, float]]] = None,
    ):
        if target_samples is None or int(target_samples) <= 0:
            raise ValueError("target_samples must be a positive integer")

        self.simulator = simulator
        self.output_dir = Path(output_dir) if output_dir is not None else next_numbered_dataset_dir()
        self.target_samples = int(target_samples)
        self.split_ratios = tuple(float(v) for v in split_ratios)
        self.seed = int(seed) if seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))
        self.save_all = bool(save_all)
        self.prefix = str(prefix)
        self.param_sets = list(param_sets) if param_sets is not None else None

        if self.param_sets is not None and len(self.param_sets) < self.target_samples:
            raise ValueError(
                "param_sets must contain at least target_samples entries when provided: "
                f"{len(self.param_sets)} < {self.target_samples}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / self.STATE_FILENAME
        self.all_path = self.output_dir / f"{self.prefix}_all.pt"

    def build(self) -> dict:
        """Run simulations until target_samples valid TraceWin samples exist."""
        state = self._load_or_create_state()
        dataset = self._load_dataset_from_state(state)

        if len(dataset) >= self.target_samples:
            saved_paths = self._save_final_outputs(dataset)
            self._write_state(state, status="complete", accepted_count=len(dataset))
            return self._summary(state, dataset, saved_paths)

        while len(dataset) < self.target_samples:
            attempt_index = int(state["attempt_index"])
            params = self._params_for_attempt(attempt_index)
            state["attempt_index"] = attempt_index + 1
            self._write_state(state, status="running", accepted_count=len(dataset))

            result = self.simulator.simulate(params)
            if not _is_valid_tracewin_result(result):
                self._write_state(state, status="running", accepted_count=len(dataset))
                continue

            x, y, score = tracewin_result_to_flat_sample(result)
            dataset.append_flat_sample(x, y, score)
            state["accepted_count"] = len(dataset)
            dataset.save_flat(self.all_path)
            self._write_state(state, status="running", accepted_count=len(dataset))

        saved_paths = self._save_final_outputs(dataset)
        self._write_state(state, status="complete", accepted_count=len(dataset))
        return self._summary(state, dataset, saved_paths)

    def _load_or_create_state(self) -> dict:
        expected = self._config_signature()
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            saved_config = state.get("config", {})
            if saved_config != expected:
                raise ValueError(
                    "Existing builder_state.json does not match the requested build "
                    f"configuration.\nExisting: {saved_config}\nRequested: {expected}"
                )
            return state

        state = {
            "version": 1,
            "status": "new",
            "attempt_index": 0,
            "accepted_count": 0,
            "config": expected,
        }
        self._write_state(state, status="new", accepted_count=0)
        return state

    def _load_dataset_from_state(self, state: dict) -> BeamDataset:
        if self.all_path.exists():
            dataset = BeamDataset.load(self.all_path)
            accepted_count = int(state.get("accepted_count", 0))
            if len(dataset) != accepted_count:
                raise ValueError(
                    f"{self.all_path} has {len(dataset)} samples but builder_state.json "
                    f"declares {accepted_count}. Fix or remove the inconsistent files."
                )
            return dataset
        if int(state.get("accepted_count", 0)) != 0:
            raise ValueError(
                f"{self.state_path} declares accepted samples, but {self.all_path} is missing."
            )
        return BeamDataset()

    def _save_final_outputs(self, dataset: BeamDataset) -> dict[str, Path]:
        return save_dataset_splits(
            dataset,
            self.output_dir,
            split=True,
            ratios=self.split_ratios,
            save_all=self.save_all,
            seed=self.seed,
            prefix=self.prefix,
        )

    def _params_for_attempt(self, attempt_index: int) -> Dict[str, float]:
        if self.param_sets is not None:
            if attempt_index >= len(self.param_sets):
                raise RuntimeError(
                    "Ran out of explicit param_sets before reaching target_samples"
                )
            return dict(self.param_sets[attempt_index])

        rng = np.random.default_rng(self.seed + int(attempt_index))
        return _sample_one_gaussian(rng)

    def _config_signature(self) -> dict:
        return {
            "target_samples": self.target_samples,
            "split_ratios": list(self.split_ratios),
            "seed": self.seed,
            "save_all": self.save_all,
            "prefix": self.prefix,
            "param_sets_count": len(self.param_sets) if self.param_sets is not None else None,
        }

    def _write_state(
        self,
        state: dict,
        *,
        status: str,
        accepted_count: int,
    ) -> None:
        state["status"] = status
        state["accepted_count"] = int(accepted_count)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _summary(
        self,
        state: dict,
        dataset: BeamDataset,
        saved_paths: dict[str, Path],
    ) -> dict:
        return {
            "output_dir": str(self.output_dir),
            "target_samples": self.target_samples,
            "n_success": len(dataset),
            "n_attempted": int(state["attempt_index"]),
            "n_failed": int(state["attempt_index"]) - len(dataset),
            "state_path": str(self.state_path),
            "paths": {name: str(path) for name, path in saved_paths.items()},
        }


def build_tracewin_dataset(
    simulator: BeamSimulator,
    output_dir: Optional[str | Path] = None,
    *,
    target_samples: Optional[int] = None,
    param_sets: Optional[Sequence[Dict[str, float]]] = None,
    n_samples: Optional[int] = None,
    split: bool = True,
    split_ratios: SplitRatios = (0.8, 0.1, 0.1),
    save_all: bool = True,
    seed: Optional[int] = 123,
    skip_failed: bool = True,
    prefix: str = "dataset",
) -> dict:
    """Run TraceWin and save a fresh/resumable dataset.

    target_samples is the preferred API. n_samples is kept as a compatibility
    alias and means "target this many valid samples".
    """
    del skip_failed  # Failed TraceWin results are never appended by the builder.

    if target_samples is None:
        if n_samples is not None:
            target_samples = int(n_samples)
        elif param_sets is not None:
            target_samples = len(param_sets)
        else:
            raise ValueError("Provide target_samples, n_samples, or param_sets")

    builder = TraceWinDatasetBuilder(
        simulator,
        output_dir,
        target_samples=int(target_samples),
        split_ratios=split_ratios,
        seed=seed,
        save_all=save_all or not split,
        prefix=prefix,
        param_sets=param_sets,
    )
    summary = builder.build()

    if not split:
        summary["paths"] = {
            name: path
            for name, path in summary["paths"].items()
            if name == "all"
        }
    return summary


def _sample_one_gaussian(rng: np.random.Generator) -> Dict[str, float]:
    defaults = default_params()
    std = dataset_std_vec()
    params = {
        key: float(defaults[key] + rng.normal(0.0, s))
        for key, s in zip(PARAM_KEYS, std)
    }
    return clip_params_to_hw(params)


def generate_param_sets_gaussian(
    n_samples: int,
    *,
    oversample_factor: float = 1.5,
    seed: Optional[int] = 0,
) -> list[Dict[str, float]]:
    """Generate parameter sets by sampling jointly from a gaussian around defaults.

    Each parameter p is drawn independently (diagonal covariance) from
    N(default_p, dataset_std_p^2), with dataset_std_p = DATASET_SCALE * sensitivity_p
    (see beam_optimization.config.adige). All parameters for a given row are
    drawn together so the returned sets reflect the actual dataset trust region
    the surrogate is trained on. Values are clipped to known hardware bounds
    (no clip where hw_min/hw_max is None).

    The total number of generated parameter sets is round(n_samples * oversample_factor);
    the extra samples absorb TraceWin failures so the builder can still reach
    n_samples valid results even if some simulations fail. Pass the returned
    list directly as param_sets to build_tracewin_dataset() or TraceWinDatasetBuilder.

    Args:
        n_samples: Target number of valid TraceWin results (used to size the output).
        oversample_factor: Generate this many times n_samples to absorb failures.
        seed: Base random seed for reproducibility.

    Returns:
        List of {param_key: value} dicts ready for param_sets=...
    """
    rng = np.random.default_rng(seed)
    defaults = np.array([p.default for p in PARAMETERS], dtype=np.float64)
    std = dataset_std_vec()
    n_total = round(n_samples * oversample_factor)

    samples = rng.normal(loc=defaults, scale=std, size=(n_total, len(PARAMETERS)))

    hw_min = np.array([p.hw_min if p.hw_min is not None else -np.inf for p in PARAMETERS], dtype=np.float64)
    hw_max = np.array([p.hw_max if p.hw_max is not None else np.inf for p in PARAMETERS], dtype=np.float64)
    samples = np.clip(samples, hw_min, hw_max)

    return [
        {key: float(val) for key, val in zip(PARAM_KEYS, row)}
        for row in samples
    ]


def _is_valid_tracewin_result(result: BeamSimulationResult) -> bool:
    return (
        result.source == "tracewin"
        and bool(result.success)
        and result.beam_states is not None
    )
