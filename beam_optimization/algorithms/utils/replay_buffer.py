"""Uniform replay buffer for off-policy algorithms (SAC, TD3).
Adapted from reinforcement_learning_2/rl/utils/replay_buffer.py.
"""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0
        self.states      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards     = np.zeros((max_size, 1),       dtype=np.float32)
        self.next_states = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.terminals   = np.zeros((max_size, 1),       dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.terminals[self.ptr]   = float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = 256):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx]),
            torch.tensor(self.actions[idx]),
            torch.tensor(self.rewards[idx]),
            torch.tensor(self.next_states[idx]),
            torch.tensor(self.terminals[idx]),
        )

    def __len__(self):
        return self.size
