"""Stable Baselines3 model-free registry, wrappers, and training integration."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np

from beam_optimization.algorithms import (
    CUSTOM_MODEL_FREE_ALGORITHMS,
    MODEL_FREE_ALGORITHMS,
    STABLE_BASELINES_ALGORITHMS,
    canonical_algorithm_name,
)
from beam_optimization.algorithms.model_free.stable_baselines import (
    StableBaselinesAgent,
)


class _ContinuousEnv(gym.Env):
    observation_space = gym.spaces.Box(
        -10.0, 10.0, shape=(6,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        -np.ones(18, dtype=np.float32),
        np.ones(18, dtype=np.float32),
        dtype=np.float32,
    )

    def __init__(self):
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(6, dtype=np.float32), {"score": 0.0}

    def step(self, action):
        self.steps += 1
        return (
            np.zeros(6, dtype=np.float32),
            -float(np.square(action).mean()),
            False,
            self.steps >= 2,
            {
                "score": float(self.steps),
                "distance_penalty": 0.0,
                "action_penalty": 0.0,
                "action_smoothness_penalty": 0.0,
                "score_regression_penalty": 0.0,
            },
        )


def _tiny_model_kwargs(name: str) -> dict:
    if name in {"sac", "td3", "ddpg"}:
        return {
            "buffer_size": 64,
            "batch_size": 4,
            "learning_starts": 1,
            "train_freq": 1,
            "gradient_steps": 1,
        }
    if name == "ppo":
        return {"n_steps": 2, "batch_size": 2, "n_epochs": 1}
    return {"n_steps": 2}


class AlgorithmNamingTests(unittest.TestCase):
    def test_public_names_are_unambiguous_and_dqn_is_excluded(self):
        self.assertEqual(
            STABLE_BASELINES_ALGORITHMS,
            ("sac", "ppo", "td3", "ddpg", "a2c"),
        )
        self.assertEqual(
            set(CUSTOM_MODEL_FREE_ALGORITHMS),
            {
                "sac_custom",
                "td3_custom",
                "ppo_custom",
                "ddpg_custom",
                "a2c_custom",
                "reinforce_custom",
                "trpo_custom",
            },
        )
        self.assertNotIn("dqn", MODEL_FREE_ALGORITHMS)
        self.assertEqual(canonical_algorithm_name("sb3_sac"), "sac")


class StableBaselinesWrapperTests(unittest.TestCase):
    def test_all_algorithms_train_save_load_and_predict_continuous_actions(self):
        with TemporaryDirectory() as tmp:
            for name in STABLE_BASELINES_ALGORITHMS:
                with self.subTest(algorithm=name):
                    env = _ContinuousEnv()
                    agent = StableBaselinesAgent(
                        name,
                        env,
                        hidden_dims=(8, 8),
                        device="cpu",
                        model_kwargs=_tiny_model_kwargs(name),
                    )
                    agent.train(n_steps=4, eval_fn=None, eval_logger=None)
                    checkpoint = Path(tmp) / f"{name}_agent"
                    agent.save(str(checkpoint))
                    zip_path = checkpoint.with_suffix(".zip")
                    self.assertTrue(zip_path.is_file())

                    loaded = StableBaselinesAgent.load(
                        name, str(zip_path), _ContinuousEnv()
                    )
                    action = loaded.select_action(np.zeros(6, dtype=np.float32))
                    self.assertEqual(action.shape, (18,))
                    self.assertTrue(np.isfinite(action).all())
                    self.assertTrue((action >= -1.0).all())
                    self.assertTrue((action <= 1.0).all())


class StableBaselinesTrainingIntegrationTests(unittest.TestCase):
    def test_reward_regularization_is_forwarded_to_training_environment(self):
        from beam_optimization.scripts import train_policies

        fake_env = SimpleNamespace()
        fake_agent = mock.Mock()
        fake_agent.train.return_value = 12.0
        with (
            TemporaryDirectory() as tmp,
            mock.patch.object(train_policies, "SurrogateEnv", return_value=fake_env) as env_cls,
            mock.patch(
                "beam_optimization.algorithms.model_free.stable_baselines."
                "StableBaselinesAgent",
                return_value=fake_agent,
            ),
        ):
            score = train_policies.train_stable_baselines(
                "ppo",
                surrogate=object(),
                dataset=object(),
                n_steps=4,
                max_ep_steps=2,
                hidden=[8, 8],
                out_dir=Path(tmp),
                enable_tensorboard=False,
                eval_episodes=0,
                enable_learning_curve=False,
                distance_penalty_weight=0.07,
                action_penalty_weight=0.05,
                action_smoothness_penalty_weight=0.25,
                score_regression_penalty_weight=2.0,
            )

        self.assertEqual(score, 12.0)
        kwargs = env_cls.call_args.kwargs
        self.assertEqual(kwargs["distance_penalty_weight"], 0.07)
        self.assertEqual(kwargs["action_penalty_weight"], 0.05)
        self.assertEqual(kwargs["action_smoothness_penalty_weight"], 0.25)
        self.assertEqual(kwargs["score_regression_penalty_weight"], 2.0)
        fake_agent.save.assert_called_once()

    def test_custom_runner_forwards_action_smoothness_weight(self):
        from beam_optimization.scripts import train_policies

        fake_env = mock.Mock()
        fake_env.observation_space.shape = (6,)
        fake_env.reset.return_value = (
            np.zeros(6, dtype=np.float32),
            {"score": 0.0},
        )
        fake_env.step.return_value = (
            np.zeros(6, dtype=np.float32),
            0.0,
            False,
            True,
            {"score": 1.0, "action_smoothness_penalty": 0.0},
        )
        fake_agent = mock.Mock()
        fake_agent.select_action.return_value = np.zeros(18, dtype=np.float32)
        fake_agent.optimize.return_value = None

        with (
            TemporaryDirectory() as tmp,
            mock.patch.object(
                train_policies,
                "SurrogateEnv",
                return_value=fake_env,
            ) as env_cls,
            mock.patch.object(
                train_policies,
                "make_custom_agent",
                return_value=fake_agent,
            ),
        ):
            train_policies.train_custom(
                "sac_custom",
                surrogate=object(),
                dataset=object(),
                n_steps=1,
                max_ep_steps=2,
                hidden=[8, 8],
                out_dir=Path(tmp),
                enable_tensorboard=False,
                eval_episodes=0,
                enable_learning_curve=False,
                action_smoothness_penalty_weight=0.25,
            )

        self.assertEqual(
            env_cls.call_args.kwargs["action_smoothness_penalty_weight"],
            0.25,
        )
        fake_agent.save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
