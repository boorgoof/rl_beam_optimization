from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from beam_optimization.env.dataset.tracewin_dataset_builder import (
    TraceWinDatasetBuilder,
)


class DatasetBuilderSeedTests(unittest.TestCase):
    def _builder(self, output_dir: Path, seed=None) -> TraceWinDatasetBuilder:
        return TraceWinDatasetBuilder(
            object(),
            output_dir=output_dir,
            target_samples=1,
            seed=seed,
        )

    def test_omitted_seed_is_generated_and_saved_in_state(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = self._builder(Path(directory) / "dataset")
            state = builder._load_or_create_state()

        self.assertIsInstance(builder.seed, int)
        self.assertGreaterEqual(builder.seed, 0)
        self.assertEqual(state["config"]["seed"], builder.seed)

    def test_resume_without_seed_reuses_saved_seed_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "dataset"
            original = self._builder(output_dir)
            original._load_or_create_state()
            original_params = original._params_for_attempt(7)

            resumed = self._builder(output_dir)
            resumed_state = resumed._load_or_create_state()

        self.assertEqual(resumed.seed, original.seed)
        self.assertEqual(resumed_state["config"]["seed"], original.seed)
        self.assertEqual(resumed._params_for_attempt(7), original_params)

    def test_explicit_seed_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = self._builder(Path(directory) / "dataset", seed=456)
            state = builder._load_or_create_state()

        self.assertEqual(builder.seed, 456)
        self.assertEqual(state["config"]["seed"], 456)


if __name__ == "__main__":
    unittest.main()
