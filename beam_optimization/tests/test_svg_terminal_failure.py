from __future__ import annotations

from contextlib import contextmanager
import unittest

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
    DifferentiableBeamState,
    DifferentiableSurrogateEnv,
)


def _state(score: float = 0.0) -> DifferentiableBeamState:
    beam0 = torch.ones((1, BEAM_STATE_DIM))
    outputs = [torch.ones((1, BEAM_STATE_DIM)) for _ in range(N_OUTPUT_STAGES)]
    return DifferentiableBeamState(
        beam0=beam0,
        params=torch.zeros(N_PARAMS),
        obs=torch.ones(BEAM_STATE_DIM * 3),
        score=torch.tensor([score]),
        beam_states=outputs,
        step_count=0,
        model_index=0,
    )


class _Simulator:
    def set_active_model(self, index):
        self.index = index


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
        env.simulator = _Simulator()
        env._action_step_t = torch.tensor(action_step_vec(), dtype=torch.float32)
        env._stage_weights_t = None
        outputs = [torch.ones((1, BEAM_STATE_DIM)) for _ in range(N_OUTPUT_STAGES)]
        outputs[-1][0, 0] = RL_MIN_NPART_RATIO - 0.001
        env._forward = lambda params, beam0: outputs

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


if __name__ == "__main__":
    unittest.main()
