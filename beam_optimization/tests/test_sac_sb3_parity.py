"""Structural parity checks between the custom SAC and SB3-SAC conventions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from beam_optimization.algorithms.model_based.mbpo import MBPO
from beam_optimization.algorithms.model_free.sac import SAC
from beam_optimization.algorithms.networks.policy_nets import GaussianPolicyNetwork


BOUNDS = ([-2.0, 10.0], [2.0, 20.0])


def make_sac(*, warmup_steps: int = 100) -> SAC:
    return SAC(
        obs_dim=3,
        act_dim=2,
        action_bounds=BOUNDS,
        hidden_dims=(16, 16),
        batch_size=256,
        warmup_steps=warmup_steps,
        device="cpu",
    )


class SACActionConventionTests(unittest.TestCase):
    def test_scale_round_trip_and_replay_storage(self):
        agent = make_sac()
        physical = np.array([2.0, 15.0], dtype=np.float32)

        scaled = agent.scale_action(physical)
        np.testing.assert_allclose(scaled, [1.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(
            agent.unscale_action(scaled),
            physical,
            atol=1e-7,
        )

        state = np.zeros(3, dtype=np.float32)
        agent.store(state, physical, 1.0, state, False)
        np.testing.assert_allclose(agent.replay.actions[0], scaled, atol=1e-7)

    def test_warmup_is_uniform_for_exactly_100_replay_steps(self):
        agent = make_sac()
        state = np.zeros(3, dtype=np.float32)
        warmup_action = np.array([1.5, 12.0], dtype=np.float32)

        agent.replay.size = 99
        with (
            patch("numpy.random.uniform", return_value=warmup_action) as uniform,
            patch.object(agent.policy, "select_action") as policy_action,
        ):
            selected = agent.select_action(state, training=True)
        np.testing.assert_allclose(selected, warmup_action)
        uniform.assert_called_once()
        policy_action.assert_not_called()

        agent.replay.size = 100
        policy_value = np.array([0.25, 14.0], dtype=np.float32)
        with (
            patch("numpy.random.uniform") as uniform,
            patch.object(agent.policy, "select_action", return_value=policy_value) as policy_action,
        ):
            selected = agent.select_action(state, training=True)
        np.testing.assert_allclose(selected, policy_value)
        uniform.assert_not_called()
        policy_action.assert_called_once()

    def test_optimize_starts_before_replay_reaches_batch_size(self):
        torch.manual_seed(7)
        np.random.seed(7)
        agent = make_sac()
        state = np.zeros(3, dtype=np.float32)
        next_state = np.ones(3, dtype=np.float32)
        for index in range(100):
            fraction = (index % 11) / 10.0
            action = np.array(
                [-2.0 + 4.0 * fraction, 10.0 + 10.0 * fraction],
                dtype=np.float32,
            )
            agent.store(state, action, 0.5, next_state, False)

        captured_actions: list[torch.Tensor] = []

        def capture_action(_module, inputs):
            captured_actions.append(inputs[1].detach().cpu())

        hooks = [
            network.register_forward_pre_hook(capture_action)
            for network in (agent.critic1, agent.critic2, agent.tc1, agent.tc2)
        ]
        try:
            losses = agent.optimize()
        finally:
            for hook in hooks:
                hook.remove()

        self.assertIsNotNone(losses)
        self.assertTrue(all(np.isfinite(value) for value in losses))
        self.assertTrue(captured_actions)
        for actions in captured_actions:
            self.assertLessEqual(float(actions.max()), 1.0 + 1e-6)
            self.assertGreaterEqual(float(actions.min()), -1.0 - 1e-6)


class SACEntropyConventionTests(unittest.TestCase):
    def test_target_entropy_and_logalpha_gradient_match_sb3_formula(self):
        agent = make_sac()
        self.assertEqual(agent.target_entropy, -2.0)
        agent.logalpha.data.fill_(-2.0)
        log_prob = torch.tensor([[-0.5], [-1.5]], dtype=torch.float32)

        loss = agent._entropy_coefficient_loss(log_prob)
        loss.backward()

        expected_gradient = -float(
            (log_prob + agent.target_entropy).mean()
        )
        self.assertAlmostEqual(
            float(agent.logalpha.grad),
            expected_gradient,
            places=6,
        )

    def test_gaussian_policy_default_keeps_physical_log_probability(self):
        policy = GaussianPolicyNetwork(3, BOUNDS, hidden_dims=(8, 8))
        state = torch.zeros((1, 3), dtype=torch.float32)

        torch.manual_seed(11)
        action_default, logp_default, *_ = policy.full_pass(state)
        torch.manual_seed(11)
        action_physical, logp_physical, *_ = policy.full_pass(
            state, include_action_scale_jacobian=True
        )
        torch.manual_seed(11)
        action_normalized, logp_normalized, *_ = policy.full_pass(
            state, include_action_scale_jacobian=False
        )

        torch.testing.assert_close(action_default, action_physical)
        torch.testing.assert_close(action_default, action_normalized)
        torch.testing.assert_close(logp_default, logp_physical)
        expected_offset = -policy._log_rescale_jacobian().sum()
        torch.testing.assert_close(
            logp_physical - logp_normalized,
            expected_offset.reshape(1, 1),
        )


class SACCheckpointCompatibilityTests(unittest.TestCase):
    def test_new_checkpoint_can_resume_and_legacy_is_inference_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            current_path = Path(temp_dir) / "current.pt"
            legacy_path = Path(temp_dir) / "legacy.pt"
            source = make_sac()
            source.store(
                np.zeros(3, dtype=np.float32),
                np.array([0.5, 12.0], dtype=np.float32),
                1.0,
                np.ones(3, dtype=np.float32),
                False,
            )
            source.save(str(current_path), include_replay=True)

            current_payload = torch.load(current_path, map_location="cpu")
            self.assertEqual(
                current_payload["implementation_version"],
                SAC.IMPLEMENTATION_VERSION,
            )
            resumed = make_sac()
            resumed.load(str(current_path), resume_training=True)
            self.assertEqual(len(resumed.replay), 1)

            current_payload.pop("implementation_version")
            current_payload.pop("action_representation")
            torch.save(current_payload, legacy_path)

            legacy_agent = make_sac()
            legacy_agent.load(str(legacy_path))
            self.assertEqual(legacy_agent.loaded_checkpoint_version, 1)
            action = legacy_agent.select_action(
                np.zeros(3, dtype=np.float32),
                training=False,
            )
            self.assertEqual(action.shape, (2,))
            with self.assertRaisesRegex(ValueError, "Legacy SAC checkpoints"):
                make_sac().load(str(legacy_path), resume_training=True)


class MBPOReplayConventionTests(unittest.TestCase):
    def test_real_and_synthetic_transitions_are_scaled_before_storage(self):
        agent = MagicMock()
        agent.scale_action.side_effect = lambda action: np.asarray(action) / 10.0
        agent.optimize.return_value = None
        agent.select_action.return_value = np.array([5.0, -5.0], dtype=np.float32)

        mixed_buffer = MagicMock()
        mixed_buffer.size = 0
        mbpo = MBPO.__new__(MBPO)
        mbpo.agent = agent
        mbpo.mixed_buffer = mixed_buffer
        mbpo.min_real_samples = 10
        mbpo.n_grad_updates = 0

        obs = np.zeros(3, dtype=np.float32)
        next_obs = np.ones(3, dtype=np.float32)
        physical = np.array([8.0, -6.0], dtype=np.float32)
        mbpo.step(obs, physical, 1.0, next_obs, False)
        stored_real = mixed_buffer.store_real.call_args.args[1]
        np.testing.assert_allclose(stored_real, [0.8, -0.6])

        synthetic_env = MagicMock()
        synthetic_env.reset.return_value = (obs, {})
        synthetic_env.step.return_value = (next_obs, 2.0, False, True, {})
        mbpo.synthetic_env = synthetic_env
        mbpo.n_synthetic_per_step = 1
        mbpo.rollout_length = 1
        mbpo._generate_synthetic()
        stored_synth = mixed_buffer.store_synth.call_args.args[1]
        np.testing.assert_allclose(stored_synth, [0.5, -0.5])


if __name__ == "__main__":
    unittest.main()
