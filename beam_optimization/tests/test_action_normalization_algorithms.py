"""Normalized-action and normalized-log-probability algorithm invariants."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from beam_optimization.algorithms.model_free.a2c import A2C
from beam_optimization.algorithms.model_free.ddpg import DDPG
from beam_optimization.algorithms.model_free.ppo import PPO
from beam_optimization.algorithms.model_free.reinforce import REINFORCE
from beam_optimization.algorithms.model_free.td3 import TD3
from beam_optimization.algorithms.model_free.trpo import TRPO
from beam_optimization.algorithms.networks.policy_nets import GaussianPolicyNetwork


BOUNDS = ([-2.0, 10.0], [2.0, 20.0])
STATE = np.zeros(3, dtype=np.float32)
NEXT_STATE = np.ones(3, dtype=np.float32)


def make_deterministic_agent(cls):
    kwargs = dict(
        obs_dim=3,
        act_dim=2,
        action_bounds=BOUNDS,
        hidden_dims=(16, 16),
        batch_size=256,
        warmup_steps=100,
        device="cpu",
    )
    return cls(**kwargs)


class DeterministicActorNormalizationTests(unittest.TestCase):
    def test_td3_and_ddpg_store_normalized_actions_and_round_trip(self):
        physical = np.array([2.0, 15.0], dtype=np.float32)
        for cls in (TD3, DDPG):
            with self.subTest(algorithm=cls.__name__):
                agent = make_deterministic_agent(cls)
                scaled = agent.scale_action(physical)
                np.testing.assert_allclose(scaled, [1.0, 0.0], atol=1e-7)
                np.testing.assert_allclose(
                    agent.unscale_action(scaled), physical, atol=1e-7
                )
                agent.store(STATE, physical, 1.0, NEXT_STATE, False)
                np.testing.assert_allclose(
                    agent.replay.actions[0], scaled, atol=1e-7
                )

    def test_td3_and_ddpg_use_exactly_100_uniform_warmup_steps(self):
        warmup_action = np.array([1.5, 12.0], dtype=np.float32)
        normalized_policy_action = np.array([0.25, -0.5], dtype=np.float32)
        for cls in (TD3, DDPG):
            with self.subTest(algorithm=cls.__name__):
                agent = make_deterministic_agent(cls)
                agent.replay.size = 99
                with (
                    patch("numpy.random.uniform", return_value=warmup_action) as uniform,
                    patch.object(agent.actor, "select_normalized_action") as actor_action,
                ):
                    selected = agent.select_action(STATE, training=True)
                np.testing.assert_allclose(selected, warmup_action)
                uniform.assert_called_once()
                actor_action.assert_not_called()

                agent.replay.size = 100
                with (
                    patch("numpy.random.uniform") as uniform,
                    patch.object(
                        agent.actor,
                        "select_normalized_action",
                        return_value=normalized_policy_action,
                    ) as actor_action,
                ):
                    selected = agent.select_action(STATE, training=False)
                np.testing.assert_allclose(
                    selected,
                    agent.unscale_action(normalized_policy_action),
                )
                uniform.assert_not_called()
                actor_action.assert_called_once()

    def test_critic_and_target_critic_receive_only_normalized_actions(self):
        for cls in (TD3, DDPG):
            with self.subTest(algorithm=cls.__name__):
                torch.manual_seed(13)
                np.random.seed(13)
                agent = make_deterministic_agent(cls)
                for index in range(100):
                    fraction = (index % 11) / 10.0
                    action = np.array(
                        [-2.0 + 4.0 * fraction, 10.0 + 10.0 * fraction],
                        dtype=np.float32,
                    )
                    agent.store(STATE, action, 0.5, NEXT_STATE, False)

                captured: list[torch.Tensor] = []

                def capture_action(_module, inputs):
                    captured.append(inputs[1].detach().cpu())

                networks = (
                    (agent.critic, agent.target_critic)
                    if cls is DDPG
                    else (
                        agent.critic.q1,
                        agent.critic.q2,
                        agent.target_critic.q1,
                        agent.target_critic.q2,
                    )
                )
                hooks = [
                    network.register_forward_pre_hook(capture_action)
                    for network in networks
                ]
                try:
                    losses = agent.optimize()
                finally:
                    for hook in hooks:
                        hook.remove()

                self.assertIsNotNone(losses)
                self.assertTrue(
                    all(value is None or np.isfinite(value) for value in losses)
                )
                self.assertTrue(captured)
                for actions in captured:
                    self.assertLessEqual(float(actions.max()), 1.0 + 1e-6)
                    self.assertGreaterEqual(float(actions.min()), -1.0 - 1e-6)


class DeterministicCheckpointCompatibilityTests(unittest.TestCase):
    def test_new_checkpoints_resume_and_legacy_checkpoints_are_inference_only(self):
        for cls in (TD3, DDPG):
            with self.subTest(algorithm=cls.__name__):
                with tempfile.TemporaryDirectory() as temp_dir:
                    current_path = Path(temp_dir) / "current.pt"
                    legacy_path = Path(temp_dir) / "legacy.pt"
                    make_deterministic_agent(cls).save(str(current_path))

                    payload = torch.load(current_path, map_location="cpu")
                    self.assertEqual(
                        payload["implementation_version"],
                        cls.IMPLEMENTATION_VERSION,
                    )
                    make_deterministic_agent(cls).load(
                        str(current_path), resume_training=True
                    )

                    payload.pop("implementation_version")
                    payload.pop("action_representation")
                    torch.save(payload, legacy_path)

                    legacy_agent = make_deterministic_agent(cls)
                    legacy_agent.load(str(legacy_path))
                    self.assertEqual(legacy_agent.loaded_checkpoint_version, 1)
                    self.assertEqual(
                        legacy_agent.select_action(STATE, training=False).shape,
                        (2,),
                    )
                    with self.assertRaisesRegex(ValueError, "Legacy"):
                        make_deterministic_agent(cls).load(
                            str(legacy_path), resume_training=True
                        )


class OnPolicyLogProbabilityTests(unittest.TestCase):
    def test_physical_jacobian_cancels_from_policy_ratios(self):
        torch.manual_seed(17)
        old_policy = GaussianPolicyNetwork(3, BOUNDS, hidden_dims=(8, 8))
        new_policy = copy.deepcopy(old_policy)
        with torch.no_grad():
            new_policy.mean_layer.bias.add_(0.15)

        states = torch.zeros((4, 3), dtype=torch.float32)
        torch.manual_seed(19)
        actions, _, _, _, _ = old_policy.full_pass(states)

        old_physical = old_policy.log_prob(states, actions)
        new_physical = new_policy.log_prob(states, actions)
        old_normalized = old_policy.log_prob(
            states, actions, include_action_scale_jacobian=False
        )
        new_normalized = new_policy.log_prob(
            states, actions, include_action_scale_jacobian=False
        )
        torch.testing.assert_close(
            new_physical - old_physical,
            new_normalized - old_normalized,
        )

    def test_normalized_reparameterized_entropy_has_finite_log_std_gradient(self):
        torch.manual_seed(23)
        policy = GaussianPolicyNetwork(3, BOUNDS, hidden_dims=(8, 8))
        states = torch.zeros((16, 3), dtype=torch.float32)
        _, log_prob, _, _, _ = policy.full_pass(
            states, include_action_scale_jacobian=False
        )
        entropy = -log_prob.mean()
        entropy.backward()

        gradient = policy.log_std_layer.bias.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_on_policy_collection_stores_normalized_log_probabilities(self):
        algorithms = (PPO, A2C, REINFORCE, TRPO)
        for cls in algorithms:
            with self.subTest(algorithm=cls.__name__):
                agent = cls(
                    obs_dim=3,
                    act_dim=2,
                    action_bounds=BOUNDS,
                    hidden_dims=(8, 8),
                    device="cpu",
                )
                torch.manual_seed(29)
                action, logged_logp, _ = agent.select_action(STATE, training=True)
                policy = getattr(agent, "policy", None) or agent.policy_network
                recomputed = policy.log_prob(
                    torch.tensor(STATE).unsqueeze(0),
                    torch.tensor(action).unsqueeze(0),
                    include_action_scale_jacobian=False,
                )
                self.assertAlmostEqual(
                    float(logged_logp),
                    float(recomputed.detach()),
                    places=5,
                )

    def test_on_policy_optimization_smoke_has_finite_losses(self):
        for cls in (PPO, A2C, REINFORCE, TRPO):
            with self.subTest(algorithm=cls.__name__):
                torch.manual_seed(31)
                np.random.seed(31)
                agent = cls(
                    obs_dim=3,
                    act_dim=2,
                    action_bounds=BOUNDS,
                    hidden_dims=(8, 8),
                    device="cpu",
                )
                for index in range(5):
                    state = np.full(3, index / 10.0, dtype=np.float32)
                    action, logpa, value = agent.select_action(
                        state, training=True
                    )
                    agent.store(
                        state,
                        action,
                        float(index + 1),
                        value,
                        float(logpa),
                        float(index == 4),
                    )
                losses = agent.optimize(last_value=0.0)
                self.assertTrue(all(np.isfinite(value) for value in losses))


if __name__ == "__main__":
    unittest.main()
