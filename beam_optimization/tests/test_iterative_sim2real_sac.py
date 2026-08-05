"""Iterative SurrogateEnv/TraceWinEnv SAC workflow tests."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from beam_optimization.algorithms.model_free.stable_baselines import (
    StableBaselinesAgent,
)
from beam_optimization.algorithms import MODEL_BASED_ALGORITHMS
from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    N_OUTPUT_STAGES,
    default_params,
    params_to_vec,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.simulation import BeamSimulationResult
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.algorithms.model_based.iterative_sim2real_sac import (
    IterativeSim2RealSAC,
    IterativeSim2RealSACConfig,
    TraceWinExperienceCollector,
    _phase_checkpoint_hook,
)


class _ContinuousEnv(gym.Env):
    def __init__(self, marker: float = 0.0):
        self.marker = marker
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.full(4, self.marker, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.full(4, self.marker, dtype=np.float32),
            0.0,
            False,
            self.steps >= 2,
            {},
        )


class _ResultEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32
        )
        beam = np.ones((N_OUTPUT_STAGES + 1, BEAM_STATE_DIM), dtype=np.float32)
        self.result = BeamSimulationResult(
            params=default_params(),
            beam_states=beam,
            score_val=1.0,
            success=True,
            source="tracewin",
        )

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {"sim_result": self.result}

    def step(self, action):
        return (
            np.zeros(1, dtype=np.float32),
            0.0,
            False,
            True,
            {"sim_result": self.result},
        )


def _tiny_dataset() -> BeamDataset:
    dataset = BeamDataset()
    beam = np.asarray(
        [1.0, 0.0, 0.0, 5.0, 5.0, 0.05, 0.05, 0.0, 0.0],
        dtype=np.float32,
    )
    x = np.concatenate([beam, params_to_vec(default_params()).astype(np.float32)])
    y = np.tile(beam, N_OUTPUT_STAGES)
    dataset.append_flat_sample(x, y, 0.0)
    return dataset


def _tiny_surrogate() -> ModularMLP:
    model = ModularMLP(
        hidden_sizes=[],
        dropout=0.0,
        latent_dim=4,
        out_hidden=[],
        out_dropout=0.0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for output in model.output_nets:
            output[-1].bias[0] = 4.0
    model.eval()
    return model


class StableBaselinesContinuationTests(unittest.TestCase):
    def test_switch_preserves_model_and_replay_can_be_reset_and_restored(self):
        env_a = _ContinuousEnv(0.0)
        env_b = _ContinuousEnv(1.0)
        agent = StableBaselinesAgent(
            "sac",
            env_a,
            hidden_dims=(8, 8),
            seed=3,
            model_kwargs={
                "learning_starts": 100,
                "buffer_size": 32,
                "batch_size": 2,
            },
        )
        model_id = id(agent._model)
        policy_id = id(agent._model.policy)
        agent.train(env_a, n_steps=2, eval_fn=None, reset_num_timesteps=True)
        self.assertGreater(agent._model.replay_buffer.size(), 0)

        with TemporaryDirectory() as tmp:
            replay = Path(tmp) / "replay.pkl"
            agent.save_replay_buffer(replay)
            agent.reset_replay_buffer()
            self.assertEqual(agent._model.replay_buffer.size(), 0)
            agent.load_replay_buffer(replay)
            self.assertGreater(agent._model.replay_buffer.size(), 0)

        agent.set_env(env_b)
        self.assertEqual(id(agent._model), model_id)
        self.assertEqual(id(agent._model.policy), policy_id)
        target = agent.delay_learning(7)
        self.assertEqual(target, agent.num_timesteps + 7)

    def test_phase_specific_sac_rate_and_gradient_steps_are_mutable(self):
        agent = StableBaselinesAgent(
            "sac",
            _ContinuousEnv(),
            hidden_dims=(8, 8),
            seed=3,
            model_kwargs={"buffer_size": 32, "batch_size": 2},
        )
        model_id = id(agent._model)
        policy_id = id(agent._model.policy)

        agent.configure_off_policy_updates(
            learning_rate=1e-5,
            gradient_steps=0,
        )
        self.assertEqual(agent._model.learning_rate, 1e-5)
        self.assertEqual(agent._model.lr_schedule(0.5), 1e-5)
        self.assertEqual(agent._model.gradient_steps, 0)
        agent.set_off_policy_gradient_steps(1)
        self.assertEqual(agent._model.gradient_steps, 1)
        self.assertEqual(id(agent._model), model_id)
        self.assertEqual(id(agent._model.policy), policy_id)


class TraceWinCollectorTests(unittest.TestCase):
    def test_reset_and_step_results_are_collected_once(self):
        updater = SimpleNamespace(
            add_tracewin_result=lambda result: result.source == "tracewin"
        )
        env = TraceWinExperienceCollector(_ResultEnv(), updater)
        env.reset()
        env.step(np.zeros(1, dtype=np.float32))
        self.assertEqual(env.accepted_reset_samples, 1)
        self.assertEqual(env.accepted_step_samples, 1)


class IterativeWorkflowTests(unittest.TestCase):
    def test_real_update_interval_enables_one_update_every_twenty_steps(self):
        updates = []
        agent = SimpleNamespace(
            set_off_policy_gradient_steps=lambda value: updates.append(value)
        )
        updater = SimpleNamespace(n_online_samples=0)
        state = {"phase_steps_completed": 0}
        with TemporaryDirectory() as tmp:
            hook = _phase_checkpoint_hook(
                output=Path(tmp),
                agent=agent,
                updater=updater,
                state=state,
                starting_steps=0,
                interval=10_000,
                budget=40,
                gradient_update_interval=20,
            )
            for step in range(1, 41):
                hook(step, step, [])

        self.assertEqual([i + 1 for i, value in enumerate(updates) if value], [20, 40])
        self.assertEqual(state["phase_steps_completed"], 40)

    def _args(self, root: Path, *, cycles: int):
        dataset_path = root / "dataset.pt"
        surrogate_path = root / "surrogate.pt"
        tracewin_path = root / "tracewin.ini"
        _tiny_dataset().save_flat(dataset_path)
        _tiny_surrogate().save(str(surrogate_path))
        tracewin_path.write_text("fake TraceWin project", encoding="utf-8")
        return IterativeSim2RealSACConfig(
            surrogate=str(surrogate_path),
            dataset=str(dataset_path),
            tracewin=str(tracewin_path),
            output=str(root / "output"),
            cycles=cycles,
            initial_surrogate_steps=2,
            subsequent_surrogate_steps=2,
            real_steps_per_cycle=2,
            real_learning_starts=1,
            max_ep_steps=2,
            online_mix_ratio=0.25,
            surrogate_update_steps=1,
            surrogate_update_batch_size=2,
            surrogate_update_lr=3e-5,
            checkpoint_every_real_steps=1,
            checkpoint_every_surrogate_steps=1,
            eval_every=1,
            eval_episodes=1,
            hidden=(8, 8),
            seed=42,
            distance_penalty_weight=0.0,
            action_penalty_weight=0.0,
            action_smoothness_penalty_weight=0.0,
            score_regression_penalty_weight=0.0,
            resume=False,
            no_tensorboard=True,
        )

    def test_two_cycles_complete_without_physical_tracewin_and_preserve_base(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, cycles=2)
            base_bytes = Path(args.surrogate).read_bytes()
            trace_model = ModularMLP.load(args.surrogate)

            class FakeTraceWinEnv(SurrogateEnv):
                def reset(self, **kwargs):
                    obs, info = super().reset(**kwargs)
                    info["sim_result"].source = "tracewin"
                    return obs, info

                def step(self, action):
                    obs, reward, terminated, truncated, info = super().step(action)
                    info["sim_result"].source = "tracewin"
                    return obs, reward, terminated, truncated, info

            def tracewin_factory(**kwargs):
                return FakeTraceWinEnv(
                    model=trace_model,
                    dataset=kwargs["distance_dataset"],
                    max_steps=kwargs["max_steps"],
                    reset_scale=kwargs["reset_scale"],
                )

            summary = IterativeSim2RealSAC(
                args, tracewin_env_factory=tracewin_factory
            ).train()

            output = Path(args.output)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["global_sac_steps"], 8)
            # Two cycles, each with one reset, two actions, and the automatic
            # reset after the two-step truncation. Surrogate evaluations must
            # not add any TraceWin calls.
            self.assertEqual(summary["online_samples"], 8)
            self.assertTrue((output / "cycle_01/updated_surrogate/surrogate_0.pt").is_file())
            self.assertTrue((output / "cycle_02/updated_surrogate/surrogate_0.pt").is_file())
            self.assertTrue((output / "sac/sac_agent.zip").is_file())
            self.assertTrue((output / "sac/replay_buffer.pkl").is_file())
            self.assertTrue((output / "sac/learning_curve.png").is_file())
            self.assertEqual(Path(args.surrogate).read_bytes(), base_bytes)

            eval_env = SurrogateEnv(
                model=ModularMLP.load(
                    str(output / "working_surrogate/surrogate_0.pt")
                ),
                dataset=BeamDataset.load(args.dataset),
                max_steps=2,
            )
            loaded = StableBaselinesAgent.load(
                "sac", str(output / "sac/sac_agent.zip"), env=eval_env
            )
            obs, _ = eval_env.reset(seed=9)
            action = loaded.select_action(obs)
            self.assertEqual(action.shape, eval_env.action_space.shape)

            resume_tracewin_calls = 0

            # A checkpoint written by the earlier workflow does not contain
            # the phase-specific tuning controls.  It must remain resumable
            # and adopt the new conservative defaults without relaxing checks
            # for any option that was already persisted.
            state_path = output / "state.json"
            legacy_state = json.loads(state_path.read_text(encoding="utf-8"))
            for key in (
                "surrogate_learning_rate",
                "real_learning_rate",
                "real_update_interval",
            ):
                legacy_state["config"].pop(key)
            state_path.write_text(json.dumps(legacy_state), encoding="utf-8")

            def forbidden_tracewin_factory(**kwargs):
                nonlocal resume_tracewin_calls
                resume_tracewin_calls += 1
                return tracewin_factory(**kwargs)

            resumed = IterativeSim2RealSAC(
                args,
                tracewin_env_factory=forbidden_tracewin_factory,
            ).resume()
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resume_tracewin_calls, 0)
            upgraded_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(upgraded_state["config"]["real_learning_rate"], 1e-5)
            self.assertEqual(upgraded_state["config"]["real_update_interval"], 20)

    def test_initial_policy_skips_initial_surrogate_and_starts_on_tracewin(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, cycles=2)
            args.enable_learning_curve = False

            dataset = BeamDataset.load(args.dataset)
            policy_env = SurrogateEnv(
                model=ModularMLP.load(args.surrogate),
                dataset=dataset,
                max_steps=args.max_ep_steps,
            )
            pretrained = StableBaselinesAgent(
                "sac",
                policy_env,
                hidden_dims=(8, 8),
                seed=7,
                model_kwargs={
                    "learning_starts": 100,
                    "buffer_size": 32,
                    "batch_size": 2,
                },
            )
            pretrained._model.num_timesteps = 7
            policy_dir = root / "pretrained_sac"
            policy_dir.mkdir()
            initial_policy = policy_dir / "sac_agent.zip"
            pretrained.save(str(initial_policy))
            policy_bytes = initial_policy.read_bytes()
            policy_env.close()
            args.initial_policy = str(policy_dir)

            surrogate_env_calls = 0

            def surrogate_factory(**kwargs):
                nonlocal surrogate_env_calls
                surrogate_env_calls += 1
                return SurrogateEnv(**kwargs)

            trace_model = ModularMLP.load(args.surrogate)

            class FakeTraceWinEnv(SurrogateEnv):
                def reset(self, **kwargs):
                    obs, info = super().reset(**kwargs)
                    info["sim_result"].source = "tracewin"
                    return obs, info

                def step(self, action):
                    obs, reward, terminated, truncated, info = super().step(action)
                    info["sim_result"].source = "tracewin"
                    return obs, reward, terminated, truncated, info

            def tracewin_factory(**kwargs):
                return FakeTraceWinEnv(
                    model=trace_model,
                    dataset=kwargs["distance_dataset"],
                    max_steps=kwargs["max_steps"],
                    reset_scale=kwargs["reset_scale"],
                )

            stdout = StringIO()
            with redirect_stdout(stdout):
                summary = IterativeSim2RealSAC(
                    args,
                    surrogate_env_factory=surrogate_factory,
                    tracewin_env_factory=tracewin_factory,
                ).train()

            # 7 checkpoint steps + cycle 1 real (2) + cycle 2 refresh (2)
            # + cycle 2 real (2). There is no initial surrogate phase.
            self.assertEqual(summary["global_sac_steps"], 13)
            self.assertEqual(summary["initial_policy"], str(policy_dir))
            self.assertEqual(surrogate_env_calls, 2)
            self.assertEqual(initial_policy.read_bytes(), policy_bytes)
            self.assertIn(
                "cycle=1/2 real_step=1/2 score=", stdout.getvalue()
            )
            state = Path(args.output, "state.json").read_text(encoding="utf-8")
            self.assertIn('"initial_policy"', state)

    def test_config_defaults_match_public_workflow(self):
        args = IterativeSim2RealSACConfig(
            surrogate="model.pt",
            dataset="dataset.pt",
            tracewin="tracewin.ini",
            output="out",
        )
        self.assertEqual(args.cycles, 1)
        self.assertEqual(args.initial_surrogate_steps, 200_000)
        self.assertEqual(args.subsequent_surrogate_steps, 20_000)
        self.assertEqual(args.real_steps_per_cycle, 2_000)
        self.assertEqual(args.real_learning_starts, 1_000)
        self.assertEqual(args.surrogate_learning_rate, 3e-4)
        self.assertEqual(args.real_learning_rate, 1e-5)
        self.assertEqual(args.real_update_interval, 20)
        self.assertEqual(args.max_ep_steps, 20)
        self.assertIsNone(args.initial_policy)
        self.assertIn("iterative_sim2real_sac", MODEL_BASED_ALGORITHMS)


class TrainPoliciesIntegrationTests(unittest.TestCase):
    def test_only_iterative_sim2real_propagates_quick_configuration(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.pt"
            surrogate_path = root / "surrogate.pt"
            tracewin_path = root / "tracewin.ini"
            output = root / "train"
            initial_policy = root / "sac_agent.zip"
            _tiny_dataset().save_flat(dataset_path)
            _tiny_surrogate().save(str(surrogate_path))
            tracewin_path.write_text("fake", encoding="utf-8")
            captured = {}

            class FakeAlgorithm:
                def __init__(self, config):
                    captured["config"] = config

                def train(self):
                    return {"best_score": 1.0, "learning_curve": []}

                def resume(self):
                    raise AssertionError("resume was not requested")

            argv = [
                "train_policies",
                "--only", "iterative_sim2real_sac",
                "--single-surrogate", str(surrogate_path),
                "--dataset", str(dataset_path),
                "--tracewin", str(tracewin_path),
                "--output", str(output),
                "--initial-policy", str(initial_policy),
                "--n-seeds", "3",
                "--quick",
                "--no-learning-curve",
                "--no-tensorboard",
                "--no-kill-stale",
            ]
            from beam_optimization.scripts import train_policies

            with patch.object(train_policies, "IterativeSim2RealSAC", FakeAlgorithm), \
                 patch("sys.argv", argv):
                train_policies.main()

            config = captured["config"]
            self.assertEqual(config.max_ep_steps, 20)
            self.assertEqual(config.initial_surrogate_steps, 200)
            self.assertEqual(config.real_steps_per_cycle, 2)
            self.assertEqual(config.real_learning_starts, 1)
            self.assertEqual(config.real_learning_rate, 1e-5)
            self.assertEqual(config.real_update_interval, 1)
            self.assertEqual(config.seed, 42)
            self.assertEqual(config.initial_policy, str(initial_policy))
            self.assertFalse(config.kill_stale)
            self.assertEqual(
                Path(config.output), output / "iterative_sim2real_sac"
            )
            self.assertTrue((output / "summary.json").is_file())

    def test_resume_requires_only_iterative_sim2real(self):
        from beam_optimization.scripts import train_policies

        with patch("sys.argv", ["train_policies", "--resume"]):
            with self.assertRaises(SystemExit):
                train_policies.main()

    def test_initial_policy_requires_only_iterative_sim2real(self):
        from beam_optimization.scripts import train_policies

        with patch(
            "sys.argv", ["train_policies", "--initial-policy", "sac_agent.zip"]
        ):
            with self.assertRaises(SystemExit):
                train_policies.main()


class PublicIdentityTests(unittest.TestCase):
    def test_benchmark_loads_iterative_checkpoint_as_sb3_sac(self):
        from beam_optimization.scripts.benchmark import make_policy_agent

        env = _ContinuousEnv()
        sentinel = object()
        with patch.object(StableBaselinesAgent, "load", return_value=sentinel) as load:
            loaded = make_policy_agent(
                "iterative_sim2real_sac", "policy.zip", env, [8, 8]
            )
        self.assertIs(loaded, sentinel)
        load.assert_called_once_with("sac", "policy.zip", env=env)

    def test_qualitative_test_loads_iterative_checkpoint_as_sb3_sac(self):
        from beam_optimization.scripts.test import make_agent

        env = _ContinuousEnv()
        sentinel = object()
        with patch.object(StableBaselinesAgent, "load", return_value=sentinel) as load:
            loaded = make_agent(
                "iterative_sim2real_sac", "policy.zip", 4, [8, 8], env=env
            )
        self.assertIs(loaded, sentinel)
        load.assert_called_once_with("sac", "policy.zip", env=env)

if __name__ == "__main__":
    unittest.main()
