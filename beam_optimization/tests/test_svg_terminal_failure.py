from __future__ import annotations

from contextlib import contextmanager
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from beam_optimization.algorithms.model_based.svg import SVGAgent
from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    ERROR_SCORE,
    N_OUTPUT_STAGES,
    N_PARAMS,
    PARAM_KEYS,
    RL_MIN_NPART_RATIO,
    TERMINAL_FAILURE_REWARD,
    action_step_vec,
)
from beam_optimization.env.surrogate_env.differentiable_surrogate_env import (
    DifferentiableEpisodeState,
    DifferentiableSurrogateEnv,
)
from beam_optimization.env.simulation import DifferentiableBeamSimulationResult


def _state(score: float = 0.0) -> DifferentiableEpisodeState:
    beam0 = torch.ones((1, BEAM_STATE_DIM))
    outputs = [torch.ones((1, BEAM_STATE_DIM)) for _ in range(N_OUTPUT_STAGES)]
    simulation = DifferentiableBeamSimulationResult(
        beam0=beam0,
        params=torch.zeros(N_PARAMS),
        beam_states=outputs,
        final_beam=outputs[-1],
        score=torch.tensor([score]),
        model_index=0,
    )
    return DifferentiableEpisodeState(
        simulation=simulation,
        obs=torch.ones(BEAM_STATE_DIM * 3),
        score=torch.tensor([score]),
        step_count=0,
    )


class _Simulator:
    def __init__(self, outputs):
        self.outputs = outputs

    def simulate_torch(self, params, beam0, model_index=None):
        return DifferentiableBeamSimulationResult(
            beam0=beam0,
            params=params,
            beam_states=self.outputs,
            final_beam=self.outputs[-1],
            score=torch.tensor(0.0),
            model_index=model_index,
        )


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.0))

    def full_pass(self, obs):
        action = self.theta.expand(1, N_PARAMS)
        logpa = self.theta.reshape(1) * 0.0
        return action, logpa, None, None, None

    def select_greedy_action(self, obs):
        return np.zeros(N_PARAMS, dtype=np.float32)


class _TerminalEnv:
    def __init__(self):
        self.calls = 0

    def reset_torch(self, beam0=None):
        return _state()

    @contextmanager
    def frozen_surrogate_weights(self):
        yield

    def step_torch(self, state, action):
        self.calls += 1
        next_state = _state(ERROR_SCORE)
        next_state.step_count = state.step_count + 1
        reward = action.sum() * 0.0 + TERMINAL_FAILURE_REWARD
        return next_state, reward, True


class SVGTerminalFailureTests(unittest.TestCase):
    def test_differentiable_step_returns_terminal_failure(self):
        env = DifferentiableSurrogateEnv.__new__(DifferentiableSurrogateEnv)
        env.device = torch.device("cpu")
        env._action_step_t = torch.tensor(action_step_vec(), dtype=torch.float32)
        env._stage_weights_t = None
        outputs = [torch.ones((1, BEAM_STATE_DIM)) for _ in range(N_OUTPUT_STAGES)]
        outputs[-1][0, 0] = RL_MIN_NPART_RATIO - 0.001
        env.simulator = _Simulator(outputs)
        env.distance_penalty_weight = 0.0
        env.action_penalty_weight = 0.0
        env.action_smoothness_penalty_weight = 0.0
        env.score_regression_penalty_weight = 0.0

        next_state, reward, terminated = env.step_torch(
            _state(),
            torch.zeros(N_PARAMS),
        )

        self.assertTrue(terminated)
        self.assertEqual(float(next_state.score), ERROR_SCORE)
        self.assertEqual(float(reward), TERMINAL_FAILURE_REWARD)

    def test_svg_training_unroll_stops_after_terminal_step(self):
        agent = SVGAgent.__new__(SVGAgent)
        agent.env = _TerminalEnv()
        agent.policy = _Policy()
        agent.optimizer = torch.optim.SGD(agent.policy.parameters(), lr=1e-3)
        agent.device = torch.device("cpu")
        agent.n_step = 20
        agent.alpha = 0.01
        agent.max_grad_norm = 1.0
        agent.best_score = 0.0
        agent.best_params = {}
        agent.param_keys = tuple(PARAM_KEYS)
        agent.train_steps = 0

        result = agent.optimize_episode()

        self.assertEqual(agent.env.calls, 1)
        self.assertEqual(result.score_history, [ERROR_SCORE])
        self.assertEqual(result.final_score, ERROR_SCORE)

    def test_svg_evaluation_unroll_stops_after_terminal_step(self):
        agent = SVGAgent.__new__(SVGAgent)
        agent.env = _TerminalEnv()
        agent.policy = _Policy()
        agent.device = torch.device("cpu")
        agent.n_step = 20

        score = agent.evaluate(n_episodes=3)

        self.assertEqual(agent.env.calls, 3)
        self.assertEqual(score, ERROR_SCORE)

    def test_svg_checkpoint_restores_best_parameter_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "svg.pt"
            source = SVGAgent.__new__(SVGAgent)
            source.policy = _Policy()
            source.optimizer = torch.optim.SGD(source.policy.parameters(), lr=1e-3)
            source.device = torch.device("cpu")
            source.best_score = 42.0
            source.best_params = {
                key: float(index)
                for index, key in enumerate(PARAM_KEYS)
            }
            source.train_steps = 7
            source.save(str(checkpoint))

            restored = SVGAgent.__new__(SVGAgent)
            restored.policy = _Policy()
            restored.optimizer = torch.optim.SGD(
                restored.policy.parameters(), lr=1e-3
            )
            restored.device = torch.device("cpu")
            restored.best_score = -float("inf")
            restored.best_params = {}
            restored.train_steps = 0
            restored.load(str(checkpoint))

            self.assertEqual(restored.best_score, source.best_score)
            self.assertEqual(restored.best_params, source.best_params)
            self.assertEqual(restored.train_steps, source.train_steps)


if __name__ == "__main__":
    unittest.main()
