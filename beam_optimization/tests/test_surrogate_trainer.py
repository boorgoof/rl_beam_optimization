"""SurrogateTrainer: per-feature loss weights, LR scheduling, early stopping."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    BEAM_STATE_FEATURES,
    N_OUTPUT_STAGES,
    N_PARAMS,
    SCORE_WEIGHTS,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.surrogate_env.surrogate.model.trainer import (
    SurrogateTrainer,
    build_feature_loss_weights,
    compute_normalization_metadata,
)


class FeatureLossWeightsTests(unittest.TestCase):
    def test_length_and_order_match_beam_state_features(self):
        weights = build_feature_loss_weights()
        self.assertEqual(tuple(weights.shape), (len(BEAM_STATE_FEATURES),))

        idx = {name: i for i, name in enumerate(BEAM_STATE_FEATURES)}
        expected_group = {
            "npart_ratio": "npart_ratio",
            "ex": "emittance", "ey": "emittance",
            "x0": "offset", "y0": "offset",
            "x'0": "angle", "y'0": "angle",
            "SizeX": "size", "SizeY": "size",
        }
        for name, group in expected_group.items():
            self.assertAlmostEqual(
                weights[idx[name]].item(), SCORE_WEIGHTS[group], msg=f"feature {name!r}"
            )


def _make_synthetic_dataset(path: Path, *, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, BEAM_STATE_DIM + N_PARAMS)).astype(np.float32)
    Y = rng.normal(size=(n, N_OUTPUT_STAGES * BEAM_STATE_DIM)).astype(np.float32)
    scores = rng.normal(size=n).astype(np.float32)
    dataset = BeamDataset()
    dataset.append_flat_samples(X, Y, scores)
    dataset.save_flat(path)


_TINY_MODEL_KWARGS = {"hidden_sizes": [8], "latent_dim": 4, "out_hidden": [8]}


class SchedulerAndEarlyStoppingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.train_path = Path(self.tmpdir.name) / "train.pt"
        self.val_path = Path(self.tmpdir.name) / "val.pt"
        _make_synthetic_dataset(self.train_path, n=16, seed=1)
        _make_synthetic_dataset(self.val_path, n=8, seed=2)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build_trainer(self, **overrides) -> SurrogateTrainer:
        kwargs = dict(
            train_dataset_path=self.train_path,
            val_dataset_path=self.val_path,
            output_dir=Path(self.tmpdir.name) / "out",
            max_epochs=50,
            batch_size=8,
            patience=3,
            model_kwargs=_TINY_MODEL_KWARGS,
            enable_tensorboard=False,
        )
        kwargs.update(overrides)
        return SurrogateTrainer(**kwargs)

    def _train_one_with_mocked_val(self, trainer: SurrogateTrainer, val_losses: list[float]):
        train_dataset = BeamDataset.load(self.train_path)
        val_dataset = BeamDataset.load(self.val_path)
        norm_stats = compute_normalization_metadata(train_dataset)
        model = ModularMLP(norm_stats=norm_stats, **trainer.model_kwargs)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=trainer.lr, weight_decay=trainer.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15,
        )
        with mock.patch.object(trainer, "_evaluate_loss", side_effect=val_losses):
            history, best_state, best_val_loss = trainer._train_one(
                model, optimizer, scheduler, train_dataset, val_dataset,
            )
        return history, best_state, best_val_loss

    def test_stops_after_patience_epochs_without_improvement(self):
        trainer = self._build_trainer(patience=3, max_epochs=50)
        # improves at epoch 2 (5.0), then plateaus for 3 straight epochs -> stop at epoch 5.
        val_losses = [10.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]
        history, _, best_val_loss = self._train_one_with_mocked_val(trainer, val_losses)

        self.assertEqual(len(history), 5)
        self.assertAlmostEqual(best_val_loss, 5.0)

    def test_patience_none_runs_full_max_epochs(self):
        trainer = self._build_trainer(patience=None, max_epochs=4)
        val_losses = [10.0, 11.0, 12.0, 13.0]  # never improves after epoch 1
        history, _, _ = self._train_one_with_mocked_val(trainer, val_losses)
        self.assertEqual(len(history), 4)

    def test_scheduler_step_called_once_per_epoch(self):
        trainer = self._build_trainer(patience=None, max_epochs=3)
        train_dataset = BeamDataset.load(self.train_path)
        val_dataset = BeamDataset.load(self.val_path)
        norm_stats = compute_normalization_metadata(train_dataset)
        model = ModularMLP(norm_stats=norm_stats, **trainer.model_kwargs)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=trainer.lr, weight_decay=trainer.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15,
        )
        with mock.patch.object(trainer, "_evaluate_loss", side_effect=[10.0, 9.0, 8.0]), \
             mock.patch.object(scheduler, "step") as mock_step:
            history, _, _ = trainer._train_one(
                model, optimizer, scheduler, train_dataset, val_dataset,
            )
        self.assertEqual(mock_step.call_count, len(history))


class FailureClassifierTrainingTests(unittest.TestCase):
    """SurrogateTrainer._train_classifier(): a single, shared classifier,
    early-stopped on validation F1 rather than loss."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.train_path = Path(self.tmpdir.name) / "train.pt"
        self.val_path = Path(self.tmpdir.name) / "val.pt"
        _make_synthetic_dataset(self.train_path, n=16, seed=1)
        _make_synthetic_dataset(self.val_path, n=8, seed=2)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build_trainer(self, **overrides) -> SurrogateTrainer:
        kwargs = dict(
            train_dataset_path=self.train_path,
            val_dataset_path=self.val_path,
            output_dir=Path(self.tmpdir.name) / "out",
            max_epochs=50,
            batch_size=8,
            n_models=2,
            patience=None,
            classifier_patience=3,
            model_kwargs=_TINY_MODEL_KWARGS,
            enable_tensorboard=False,
        )
        kwargs.update(overrides)
        return SurrogateTrainer(**kwargs)

    def _train_classifier_with_mocked_val(self, trainer, val_results):
        train_dataset = BeamDataset.load(self.train_path)
        val_dataset = BeamDataset.load(self.val_path)
        norm_stats = compute_normalization_metadata(train_dataset)
        with mock.patch.object(trainer, "_evaluate_classifier", side_effect=val_results):
            return trainer._train_classifier(train_dataset, val_dataset, norm_stats)

    def test_stops_after_patience_epochs_without_f1_improvement(self):
        trainer = self._build_trainer(classifier_patience=3)
        # improves at epoch 2 (f1=0.6), then plateaus at 0.5 for 3 straight
        # epochs -> stop at epoch 5.
        val_results = [
            (10.0, {"precision": 0.4, "recall": 0.4, "f1": 0.4}),
            (8.0, {"precision": 0.6, "recall": 0.6, "f1": 0.6}),
            (8.0, {"precision": 0.5, "recall": 0.5, "f1": 0.5}),
            (8.0, {"precision": 0.5, "recall": 0.5, "f1": 0.5}),
            (8.0, {"precision": 0.5, "recall": 0.5, "f1": 0.5}),
            (8.0, {"precision": 0.5, "recall": 0.5, "f1": 0.5}),
            (8.0, {"precision": 0.5, "recall": 0.5, "f1": 0.5}),
        ]
        _, history, _ = self._train_classifier_with_mocked_val(trainer, val_results)

        self.assertEqual(len(history), 5)

    def test_classifier_patience_none_runs_full_max_epochs(self):
        trainer = self._build_trainer(classifier_patience=None, max_epochs=4)
        val_results = [
            (10.0, {"precision": 0.4, "recall": 0.4, "f1": 0.4}),
            (11.0, {"precision": 0.4, "recall": 0.4, "f1": 0.4}),
            (12.0, {"precision": 0.4, "recall": 0.4, "f1": 0.4}),
            (13.0, {"precision": 0.4, "recall": 0.4, "f1": 0.4}),
        ]
        _, history, _ = self._train_classifier_with_mocked_val(trainer, val_results)

        self.assertEqual(len(history), 4)

    def test_train_saves_exactly_one_shared_classifier_regardless_of_n_models(self):
        trainer = self._build_trainer(n_models=2, max_epochs=2, classifier_patience=None)

        summary = trainer.train()

        classifier_files = list(trainer.output_dir.glob("failure_classifier_*.pt"))
        surrogate_files = list(trainer.output_dir.glob("surrogate_*.pt"))
        self.assertEqual(len(classifier_files), 1)
        self.assertEqual(len(surrogate_files), 2)
        self.assertEqual(summary["classifier"]["path"], str(classifier_files[0]))

    def test_skip_classifier_trains_no_classifier_file(self):
        trainer = self._build_trainer(n_models=1, max_epochs=2, train_classifier=False)

        summary = trainer.train()

        self.assertNotIn("classifier", summary)
        self.assertEqual(list(trainer.output_dir.glob("failure_classifier_*.pt")), [])


if __name__ == "__main__":
    unittest.main()
