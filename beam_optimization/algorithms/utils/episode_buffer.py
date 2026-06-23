"""On-policy trajectory buffer with GAE advantage computation (for PPO).
Adapted from reinforcement_learning_2/rl/utils/episode_buffer.py.
"""
import numpy as np
import torch


class EpisodeBuffer:
    """Stores one or more full episodes; computes discounted returns and GAE."""

    def __init__(self, gamma: float = 0.99, tau: float = 0.95):
        self.gamma = gamma
        self.tau   = tau
        self._reset()

    def _reset(self):
        self._states    = []
        self._actions   = []
        self._rewards   = []
        self._values    = []
        self._logpas    = []
        self._terminals = []

    def store(self, state, action, reward, value, logpa, done):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._logpas.append(logpa)
        self._terminals.append(done)

    def get(self, last_value: float = 0.0):
        """Compute returns + GAE, return tensors, reset buffer."""
        states  = np.array(self._states,  dtype=np.float32)
        actions = np.array(self._actions, dtype=np.float32)
        rewards = np.array(self._rewards, dtype=np.float32)
        values  = np.array(self._values,  dtype=np.float32)
        logpas  = np.array(self._logpas,  dtype=np.float32)
        T = len(rewards)

        returns = np.zeros(T, dtype=np.float32)
        G = last_value
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G * (1.0 - self._terminals[t])
            returns[t] = G

        values_ext = np.append(values, last_value)
        gaes = np.zeros(T, dtype=np.float32)
        gae  = 0.0
        for t in reversed(range(T)):
            not_done = 1.0 - self._terminals[t]
            delta = rewards[t] + self.gamma * values_ext[t + 1] * not_done - values[t]
            gae   = delta + self.gamma * self.tau * not_done * gae
            gaes[t] = gae

        self._reset()
        return (
            torch.tensor(states),
            torch.tensor(actions),
            torch.tensor(returns),
            torch.tensor(gaes),
            torch.tensor(logpas),
        )
