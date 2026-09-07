"""SurrogateTrainer: per-feature loss weights, LR scheduling, early stopping."""
from __future__ import annotations

import math
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
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.surrogate_env.surrogate.model.updater import (
    SurrogateDatasetUpdater,
)
from beam_optimization.env.surrogate_env.surrogate.model.trainer import (
    MIN_LOSS_FEATURE_STD,
    SurrogateTrainer,
    _inverse_softplus,
    _weighted_standardized_mse,
    build_feature_loss_weights,
    compute_normalization_metadata,
    compute_stage_feature_stds,
)


class FeatureLossWeightsTests(unittest.TestCase):
    def test_length_and_order_match_beam_state_features(self):
        weights = build_feature_loss_weights()
        self.assertEqual(tuple(weights.shape), (len(BEAM_STATE_FEATURES),))
        torch.testing.assert_close(weights, torch.ones(len(BEAM_STATE_FEATURES)))

    def test_standardized_loss_uses_feature_scales_before_priorities(self):
        prediction = torch.tensor([[2.0, 20.0]])
        target = torch.zeros_like(prediction)
        feature_std = torch.tensor([2.0, 20.0])
        priorities = torch.tensor([1.0, 4.0])

        loss = _weighted_standardized_mse(
            prediction, target, priorities, feature_std
        )

        self.assertAlmostEqual(loss.item(), 2.5)

    def test_standardized_loss_clamps_near_zero_feature_scale(self):
        prediction = torch.tensor([[MIN_LOSS_FEATURE_STD]])
        target = torch.zeros_like(prediction)
        loss = _weighted_standardized_mse(
            prediction,
            target,
            torch.ones(1),
            torch.zeros(1),
        )
        self.assertAlmostEqual(loss.item(), 1.0)


def _dataset_with_known_npart_ratio(*, n: int = 5, seed: int = 0):
    """Synthetic dataset where every feature except npart_ratio is arbitrary,
    beam0's npart_ratio is always exactly 1.0 (matches the real dataset
    invariant -- see visualize_surrogate_model.ipynb section 3.4), and each
    output stage's npart_ratio is a known, controlled value in (0, 1) so
    compute_normalization_metadata()'s logit transform can be checked exactly.
    """
    rng = np.random.default_rng(seed)
    npart_idx = BEAM_STATE_FEATURES.index("npart_ratio")

    X = rng.normal(size=(n, BEAM_STATE_DIM + N_PARAMS)).astype(np.float32)
    X[:, npart_idx] = 1.0

    Y = rng.normal(size=(n, N_OUTPUT_STAGES * BEAM_STATE_DIM)).astype(np.float32)
    known_npart_ratios = np.linspace(0.02, 0.9, n).astype(np.float32)
    for stage in range(N_OUTPUT_STAGES):
        col = stage * BEAM_STATE_DIM + npart_idx
        Y[:, col] = known_npart_ratios

    scores = rng.normal(size=n).astype(np.float32)
    dataset = BeamDataset()
    dataset.append_flat_samples(X, Y, scores)
    return dataset, known_npart_ratios


class ComputeNormalizationMetadataLogitTests(unittest.TestCase):
    def test_output_stage_npart_ratio_uses_logit_space_stats(self):
        dataset, known = _dataset_with_known_npart_ratio()
        npart_idx = BEAM_STATE_FEATURES.index("npart_ratio")

        norm_stats = compute_normalization_metadata(dataset)

        expected_logit = torch.logit(torch.tensor(known), eps=1e-4)
        expected_mean = expected_logit.mean().item()
        expected_var = expected_logit.var(unbiased=False).item()

        for stage_idx in range(1, N_OUTPUT_STAGES + 1):
            mean = norm_stats["beam_state_means"][stage_idx][npart_idx].item()
            var = norm_stats["beam_state_variances"][stage_idx][npart_idx].item()
            self.assertAlmostEqual(mean, expected_mean, places=4, msg=f"stage {stage_idx}")
            self.assertAlmostEqual(var, expected_var, places=4, msg=f"stage {stage_idx}")

    def test_stage_zero_beam0_npart_ratio_stays_raw(self):
        # beam0's npart_ratio is a constant 1.0; it must stay untransformed
        # (raw mean 1.0, raw variance 0.0), not logit-transformed (which
        # would be +inf without the eps clamp -- compute_normalization_metadata()
        # deliberately skips stage 0).
        dataset, _ = _dataset_with_known_npart_ratio()
        npart_idx = BEAM_STATE_FEATURES.index("npart_ratio")

        norm_stats = compute_normalization_metadata(dataset)

        self.assertAlmostEqual(norm_stats["beam_state_means"][0][npart_idx].item(), 1.0, places=5)
        self.assertAlmostEqual(norm_stats["beam_state_variances"][0][npart_idx].item(), 0.0, places=5)

    def test_loss_feature_std_uses_raw_npart_ratio(self):
        dataset, known = _dataset_with_known_npart_ratio()
        npart_idx = BEAM_STATE_FEATURES.index("npart_ratio")

        stage_stds = compute_stage_feature_stds(dataset)
        expected = torch.tensor(known).std(unbiased=False).item()

        for stage_idx, feature_std in enumerate(stage_stds, start=1):
            self.assertAlmostEqual(
                feature_std[npart_idx].item(),
                expected,
                places=5,
                msg=f"stage {stage_idx}",
            )

    def test_signed_features_remain_raw_at_every_stage(self):
        dataset, _ = _dataset_with_known_npart_ratio()
        transformed_features = {"npart_ratio", "SizeX", "SizeY", "ex", "ey"}
        _, beam_states = dataset.get_training_batch(np.arange(len(dataset)))

        norm_stats = compute_normalization_metadata(dataset)

        for stage_idx, tensor in enumerate(beam_states):
            for feature_idx, feature in enumerate(BEAM_STATE_FEATURES):
                if feature in transformed_features:
                    continue
                expected_mean = tensor[:, feature_idx].mean().item()
                mean = norm_stats["beam_state_means"][stage_idx][feature_idx].item()
                self.assertAlmostEqual(
                    mean, expected_mean, places=4,
                    msg=f"stage {stage_idx} feature {feature_idx}",
                )

    def test_positive_output_features_use_inverse_softplus_stats(self):
        dataset, _ = _dataset_with_known_npart_ratio()
        _, beam_states = dataset.get_training_batch(np.arange(len(dataset)))
        norm_stats = compute_normalization_metadata(dataset)

        for stage_idx, tensor in enumerate(beam_states[1:], start=1):
            for feature in ("SizeX", "SizeY", "ex", "ey"):
                feature_idx = BEAM_STATE_FEATURES.index(feature)
                transformed = _inverse_softplus(tensor[:, feature_idx])
                expected_mean = transformed.mean().item()
                expected_var = transformed.var(unbiased=False).item()
                self.assertAlmostEqual(
                    norm_stats["beam_state_means"][stage_idx][feature_idx].item(),
                    expected_mean,
                    places=4,
                )
                self.assertAlmostEqual(
                    norm_stats["beam_state_variances"][stage_idx][feature_idx].item(),
                    expected_var,
                    places=4,
                )
                reconstructed = torch.nn.functional.softplus(transformed)
                torch.testing.assert_close(
                    reconstructed,
                    tensor[:, feature_idx].clamp_min(MIN_LOSS_FEATURE_STD),
                )


class OnlineFineTuningLossTests(unittest.TestCase):
    def test_updater_uses_same_stage_standardized_weighted_loss(self):
        dataset, _ = _dataset_with_known_npart_ratio(n=8)
        model = ModularMLP(
            norm_stats=compute_normalization_metadata(dataset),
            **_TINY_MODEL_KWARGS,
        )
        updater = SurrogateDatasetUpdater(
            model,
            offline_dataset=dataset,
            initial_online_dataset=dataset,
            batch_size=4,
            epochs=1,
            min_samples=1,
            online_mix_ratio=0.5,
            device="cpu",
        )

        with mock.patch(
            "beam_optimization.env.surrogate_env.surrogate.model.updater."
            "_weighted_standardized_mse",
            wraps=_weighted_standardized_mse,
        ) as standardized_loss:
            losses = updater.update()

        self.assertEqual(standardized_loss.call_count, N_OUTPUT_STAGES)
        self.assertTrue(math.isfinite(losses["surrogate_0"]))
        self.assertEqual(len(updater._last_loss_stds), N_OUTPUT_STAGES)


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

    def test_train_saves_only_requested_surrogate_checkpoints(self):
        trainer = self._build_trainer(n_models=2, max_epochs=1, patience=None)

        summary = trainer.train()

        saved_files = sorted(trainer.output_dir.glob("*.pt"))
        self.assertEqual(len(saved_files), 2)
        self.assertTrue(all(path.name.startswith("surrogate_") for path in saved_files))
        self.assertEqual(len(summary["checkpoints"]), 2)
        self.assertEqual(set(summary), {"output_dir", "n_models", "checkpoints"})




if __name__ == "__main__":
    unittest.main()
