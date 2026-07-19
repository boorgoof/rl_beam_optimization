"""Build flat .pt datasets by running TraceWin simulations.

The main entry point is TraceWinDatasetBuilder. It is designed for expensive
offline TraceWin generation: it keeps an incremental dataset_all.pt and a
builder_state.json file so an interrupted run can be resumed until the target
number of valid samples is reached.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

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
        seed: Optional[int] = None,
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
        self.seed = self._resolve_seed(seed)

    def _resolve_seed(self, requested_seed: Optional[int]) -> int:
        """Resolve an explicit, resumed, or newly generated sampling seed."""
        if requested_seed is not None:
            return int(requested_seed)
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            saved_seed = state.get("config", {}).get("seed")
            if saved_seed is not None:
                return int(saved_seed)
        return int(np.random.default_rng().integers(0, 2**31 - 1))

    def build(self) -> dict:
        """Run simulations until target_samples valid TraceWin samples exist."""
        state = self._load_or_create_state()
        dataset = self._load_dataset_from_state(state)

        if len(dataset) >= self.target_samples:
            saved_paths = self._save_final_outputs(dataset)
            self._write_state(state, status="complete", accepted_count=len(dataset))
            return self._summary(state, dataset, saved_paths)

        if int(state["attempt_index"]) > 0:
            print(
                f"Resuming {self.output_dir}: {len(dataset)}/{self.target_samples} "
                f"accepted so far, {state['attempt_index']} attempts already made.",
                flush=True,
            )

        while len(dataset) < self.target_samples:
            attempt_index = int(state["attempt_index"])
            params = self._params_for_attempt(attempt_index)
            state["attempt_index"] = attempt_index + 1
            self._write_state(state, status="running", accepted_count=len(dataset))

            result = self.simulator.simulate(params)
            valid = _is_valid_tracewin_result(result)
            print(
                f"  [attempt {attempt_index + 1}] "
                f"{'accepted' if valid else 'rejected'} score={result.score_val:.6g} "
                f"-> {len(dataset) + (1 if valid else 0)}/{self.target_samples} accepted"
                + ("" if valid else f" ({result.error})"),
                flush=True,
            )
            if not valid:
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
            if "sampling" not in saved_config:
                # Legacy state written before the sampling block existed: adopt
                # the current distribution (there is no recorded one to check
                # against) and persist it for future resumes.
                saved_config = {**saved_config, "sampling": expected["sampling"]}
                state["config"] = saved_config
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
        # The sampling block pins the gaussian distribution the samples are
        # drawn from (defaults, DATASET_SCALE*sensitivity, hardware clip).
        # Without it, resuming after a change to adige.py (e.g. recalibrated
        # sensitivities or a new DATASET_SCALE) would silently mix two
        # different distributions in the same dataset file.
        if self.param_sets is not None:
            sampling = None  # explicit param_sets bypass gaussian sampling
        else:
            defaults = default_params()
            sampling = {
                "defaults": [float(defaults[key]) for key in PARAM_KEYS],
                "std": [float(value) for value in dataset_std_vec()],
                "hw_min": [p.hw_min for p in PARAMETERS],
                "hw_max": [p.hw_max for p in PARAMETERS],
            }
        return {
            "target_samples": self.target_samples,
            "split_ratios": list(self.split_ratios),
            "seed": self.seed,
            "save_all": self.save_all,
            "prefix": self.prefix,
            "param_sets_count": len(self.param_sets) if self.param_sets is not None else None,
            "sampling": sampling,
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


def _sample_one_gaussian(rng: np.random.Generator) -> Dict[str, float]:
    defaults = default_params()
    std = dataset_std_vec()
    params = {
        key: float(defaults[key] + rng.normal(0.0, s))
        for key, s in zip(PARAM_KEYS, std)
    }
    return clip_params_to_hw(params)


def _is_valid_tracewin_result(result: BeamSimulationResult) -> bool:
    return (
        result.source == "tracewin"
        and bool(result.success)
        and result.beam_states is not None
    )
