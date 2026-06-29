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


class MixedReplayBuffer:
    """Samples from real and synthetic buffers with a fixed real/synth ratio.

    Args:
        obs_dim:    Observation dimension.
        act_dim:    Action dimension.
        real_size:  Max real transitions.
        synth_size: Max synthetic transitions.
        real_ratio: Fraction of each mini-batch drawn from real_buffer.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        real_size: int = int(1e5),
        synth_size: int = int(1e6),
        real_ratio: float = 0.05,
    ):
        self.real_buffer  = ReplayBuffer(obs_dim, act_dim, real_size)
        self.synth_buffer = ReplayBuffer(obs_dim, act_dim, synth_size)
        self.real_ratio   = float(real_ratio)

    def store_real(self, s, a, r, ns, done):
        self.real_buffer.store(s, a, r, ns, float(done))

    def store_synth(self, s, a, r, ns, done):
        self.synth_buffer.store(s, a, r, ns, float(done))

    def store(self, s, a, r, ns, done):
        self.store_real(s, a, r, ns, float(done))

    def sample(self, batch_size: int):
        n_real  = max(1, int(batch_size * self.real_ratio))
        n_synth = batch_size - n_real

        if len(self.synth_buffer) < n_synth:
            return self.real_buffer.sample(batch_size)

        real_batch  = self.real_buffer.sample(n_real)
        synth_batch = self.synth_buffer.sample(n_synth)
        return tuple(torch.cat([r, s], dim=0) for r, s in zip(real_batch, synth_batch))

    def __len__(self) -> int:
        return len(self.real_buffer) + len(self.synth_buffer)

    @property
    def size(self) -> int:
        return len(self.real_buffer)
