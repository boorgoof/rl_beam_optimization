from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from beam_optimization.main import COMMAND_MODULES
from beam_optimization.config.adige import BEAM_STATE_DIM, N_OUTPUT_STAGES, N_PARAMS
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.dataset.merge_datasets import merge_dataset_files


class MergeDatasetsTests(unittest.TestCase):
    def _dataset(self, path: Path, start: int, count: int) -> None:
        X = torch.arange(
            start,
            start + count * (BEAM_STATE_DIM + N_PARAMS),
            dtype=torch.float32,
        ).reshape(count, BEAM_STATE_DIM + N_PARAMS)
        Y = torch.arange(
            start + 1000,
            start + 1000 + count * N_OUTPUT_STAGES * BEAM_STATE_DIM,
            dtype=torch.float32,
        ).reshape(count, N_OUTPUT_STAGES * BEAM_STATE_DIM)
        dataset = BeamDataset()
        dataset.append_flat_samples(X, Y, torch.full((count,), -123.0))
        dataset.save_flat(path)

    def test_merges_in_order_recalculates_scores_and_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.pt"
            second = root / "second.pt"
            self._dataset(first, 0, 7)
            self._dataset(second, 10_000, 3)

            summary = merge_dataset_files([first, second], root / "merged", seed=77)
            merged = BeamDataset.load(root / "merged" / "dataset_all.pt")
            original_first = BeamDataset.load(first)
            original_second = BeamDataset.load(second)

            self.assertEqual(summary["num_samples"], 10)
            self.assertEqual(summary["split_counts"], {"all": 10, "train": 8, "val": 1, "test": 1})
            self.assertTrue(torch.equal(merged.X[:7], original_first.X))
            self.assertTrue(torch.equal(merged.X[7:], original_second.X))
            self.assertFalse(torch.all(merged.scores == -123.0))

    def test_split_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.pt"
            second = root / "second.pt"
            self._dataset(first, 0, 10)
            self._dataset(second, 10_000, 10)
            merge_dataset_files([first, second], root / "one", seed=5)
            merge_dataset_files([first, second], root / "two", seed=5)

            for split in ("train", "val", "test"):
                one = BeamDataset.load(root / "one" / f"dataset_{split}.pt")
                two = BeamDataset.load(root / "two" / f"dataset_{split}.pt")
                self.assertTrue(torch.equal(one.X, two.X))
                self.assertTrue(torch.equal(one.Y, two.Y))

    def test_rejects_incompatible_metadata_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.pt"
            bad_metadata = root / "bad_metadata.pt"
            bad_values = root / "bad_values.pt"
            self._dataset(good, 0, 2)

            raw = torch.load(good, weights_only=False)
            raw["x_cols"] = list(raw["x_cols"])
            raw["x_cols"][0] = "wrong"
            torch.save(raw, bad_metadata)
            with self.assertRaisesRegex(ValueError, "metadata 'x_cols'"):
                merge_dataset_files([good, bad_metadata], root / "out_metadata")

            raw = torch.load(good, weights_only=False)
            raw["Y"] = raw["Y"].clone()
            raw["Y"][0, 0] = float("nan")
            torch.save(raw, bad_values)
            with self.assertRaisesRegex(ValueError, "contains NaN or infinite"):
                merge_dataset_files([good, bad_values], root / "out_values")

    def test_rejects_wrong_shapes_and_score_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.pt"
            bad_shape = root / "bad_shape.pt"
            bad_scores = root / "bad_scores.pt"
            self._dataset(good, 0, 2)

            raw = torch.load(good, weights_only=False)
            raw["X"] = raw["X"][:, :-1]
            torch.save(raw, bad_shape)
            with self.assertRaisesRegex(ValueError, "X must have shape"):
                merge_dataset_files([good, bad_shape], root / "out_shape")

            raw = torch.load(good, weights_only=False)
            raw["scores"] = raw["scores"][:1]
            torch.save(raw, bad_scores)
            with self.assertRaisesRegex(ValueError, "different sample counts"):
                merge_dataset_files([good, bad_scores], root / "out_scores")

    def test_rejects_existing_output_and_running_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_dir = root / "001"
            first_dir.mkdir()
            first = first_dir / "dataset_all.pt"
            second = root / "second.pt"
            self._dataset(first, 0, 2)
            self._dataset(second, 1000, 2)

            (first_dir / "builder_state.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "still running"):
                merge_dataset_files([first, second], root / "out")

            (first_dir / "builder_state.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            existing_dir = root / "existing"
            existing_dir.mkdir()
            (existing_dir / "dataset_all.pt").touch()
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                merge_dataset_files([first, second], existing_dir)

    def test_requires_two_unique_inputs_and_rejects_output_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.pt"
            second = root / "second.pt"
            self._dataset(first, 0, 2)
            self._dataset(second, 1000, 2)

            with self.assertRaisesRegex(ValueError, "At least two"):
                merge_dataset_files([first], root / "out")
            with self.assertRaisesRegex(ValueError, "specified more than once"):
                merge_dataset_files([first, first], root / "out")
            collision_dir = first.parent
            collision = collision_dir / "dataset_all.pt"
            self._dataset(collision, 2000, 2)
            with self.assertRaisesRegex(ValueError, "also be an output"):
                merge_dataset_files([collision, second], collision_dir)

    def test_cli_is_registered(self):
        self.assertEqual(
            COMMAND_MODULES["merge_datasets"],
            "beam_optimization.scripts.merge_datasets",
        )


if __name__ == "__main__":
    unittest.main()
