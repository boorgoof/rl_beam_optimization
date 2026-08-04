"""Tests for the dedicated real-TraceWin MBPO configuration."""

from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from beam_optimization.algorithms.model_based.mbpo import MBPO
from beam_optimization.algorithms.model_based.mbpo_model_update import (
    MBPOWithModelUpdate,
)
from beam_optimization.scripts import train_policies


class _CountingBuffer:
    def __init__(self):
        self.size = 0

    def store_real(self, *_args):
        self.size += 1


class _Agent:
    @staticmethod
    def scale_action(action):
        return action

    @staticmethod
    def optimize():
        return None


class MBPOTraceWinConfigurationTests(unittest.TestCase):
    def test_mbpo_constructor_defaults_remain_legacy(self):
        mbpo_defaults = inspect.signature(MBPO.__init__).parameters
        update_defaults = inspect.signature(
            MBPOWithModelUpdate.__init__
        ).parameters

        self.assertEqual(mbpo_defaults["n_synthetic_per_step"].default, 40)
        self.assertEqual(mbpo_defaults["min_real_samples"].default, 256)
        self.assertEqual(update_defaults["model_train_freq"].default, 50)

    def test_step_21_through_270_produce_300000_nominal_transitions(self):
        mbpo = MBPO.__new__(MBPO)
        mbpo.agent = _Agent()
        mbpo.mixed_buffer = _CountingBuffer()
        mbpo.min_real_samples = 21
        mbpo.n_grad_updates = 1
        mbpo.n_synthetic_per_step = 40
        mbpo.rollout_length = 30
        mbpo._generate_synthetic = mock.Mock()

        transition = np.zeros(1, dtype=np.float32)
        for _ in range(270):
            mbpo.step(transition, transition, 0.0, transition, False)

        self.assertEqual(mbpo._generate_synthetic.call_count, 250)
        self.assertEqual(
            mbpo._generate_synthetic.call_count
            * mbpo.n_synthetic_per_step
            * mbpo.rollout_length,
            300_000,
        )

    def test_periodic_update_at_200_and_final_update_at_270(self):
        mbpo = MBPOWithModelUpdate.__new__(MBPOWithModelUpdate)
        mbpo.model_train_freq = 200
        mbpo._real_step_count = 0
        mbpo._last_model_update_step = 0
        mbpo.last_update_losses = None
        mbpo.dataset_save_path = None
        mbpo.surrogate_save_dir = None
        mbpo._updater = mock.Mock()
        mbpo._updater.update_if_ready.side_effect = [
            {"surrogate_0": 2.0},
            {"surrogate_0": 1.0},
        ]

        transition = np.zeros(1, dtype=np.float32)
        with mock.patch.object(MBPO, "step", return_value=None):
            for _ in range(270):
                mbpo.step(transition, transition, 0.0, transition, False)
            mbpo.finalize_model_update()

        self.assertEqual(mbpo._updater.update_if_ready.call_count, 2)
        self.assertEqual(mbpo._last_model_update_step, 270)
        self.assertEqual(mbpo.last_update_losses, {"surrogate_0": 1.0})

    def test_final_update_is_not_duplicated_on_period_boundary(self):
        mbpo = MBPOWithModelUpdate.__new__(MBPOWithModelUpdate)
        mbpo.model_train_freq = 200
        mbpo._real_step_count = 0
        mbpo._last_model_update_step = 0
        mbpo.last_update_losses = None
        mbpo.dataset_save_path = None
        mbpo.surrogate_save_dir = None
        mbpo._updater = mock.Mock()
        mbpo._updater.update_if_ready.return_value = {"surrogate_0": 1.0}

        transition = np.zeros(1, dtype=np.float32)
        with mock.patch.object(MBPO, "step", return_value=None):
            for _ in range(200):
                mbpo.step(transition, transition, 0.0, transition, False)
            mbpo.finalize_model_update()

        self.assertEqual(mbpo._updater.update_if_ready.call_count, 1)

    def test_working_ensemble_copy_is_canonical_and_base_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            updated = root / "updated"
            base.mkdir()
            original = {
                "surrogate_018_0.pt": b"model-zero",
                "surrogate_018_1.pt": b"model-one",
            }
            for name, payload in original.items():
                (base / name).write_bytes(payload)

            train_policies.initialize_updated_ensemble_from_base(base, updated)

            self.assertEqual(
                sorted(path.name for path in updated.glob("surrogate_*.pt")),
                ["surrogate_0.pt", "surrogate_1.pt"],
            )
            self.assertEqual((updated / "surrogate_0.pt").read_bytes(), b"model-zero")
            self.assertEqual((updated / "surrogate_1.pt").read_bytes(), b"model-one")
            for name, payload in original.items():
                self.assertEqual((base / name).read_bytes(), payload)

    def test_cli_propagates_mbpo_parameters_and_disables_evaluation(self):
        captured = {}

        def fake_train_dyna(*_args, **kwargs):
            captured.update(kwargs)
            return 0.0

        def fake_run_seeded(_label, _root, seeds, _files, train_fn, _curves):
            train_fn(seeds[0], Path("/tmp/mbpo-test"), {})
            return {"best_score_mean": 0.0, "best_score_std": 0.0}

        argv = [
            "train_policies",
            "--base-ensemble", "base",
            "--updated-ensemble", "updated",
            "--dataset", "dataset.pt",
            "--output", "output",
            "--tracewin", "tracewin.ini",
            "--online-finetune",
            "--n-synthetic-per-step", "40",
            "--mbpo-min-real-samples", "21",
            "--model-train-freq", "200",
            "--no-learning-curve",
            "--skip",
            "sac", "ppo", "td3", "ddpg", "a2c",
            "sac_custom", "td3_custom", "ppo_custom", "ddpg_custom",
            "a2c_custom", "reinforce_custom", "trpo_custom", "svg",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                train_policies, "initialize_updated_ensemble_from_base"
            ),
            mock.patch.object(
                train_policies, "load_surrogate_ensemble", return_value=["model"]
            ),
            mock.patch.object(
                train_policies.BeamDataset, "load", return_value=object()
            ),
            mock.patch.object(train_policies, "train_dyna", side_effect=fake_train_dyna),
            mock.patch.object(train_policies, "run_seeded", side_effect=fake_run_seeded),
            mock.patch.object(train_policies, "print_summary"),
            mock.patch.object(
                train_policies, "save_all_learning_curves_plot", return_value=None
            ),
        ):
            train_policies.main()

        self.assertEqual(captured["n_synthetic_per_step"], 40)
        self.assertEqual(captured["mbpo_min_real_samples"], 21)
        self.assertEqual(captured["model_train_freq"], 200)
        self.assertFalse(captured["enable_learning_curve"])


if __name__ == "__main__":
    unittest.main()
