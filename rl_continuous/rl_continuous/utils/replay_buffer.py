import numpy as np
import torch


class ReplayBuffer:
    """
    ReplayBuffer stores transitions (s, a, r, s', done) collected during environment interaction.
    It is used by all off-policy algorithms: DDPG, TD3, SAC.
    Samples are drawn uniformly at random — a simple but effective decorrelation technique
    introduced in DQN (Mnih et al., 2015) and reused in all actor-critic off-policy methods.
    """
    def __init__(self, obs_dim, act_dim, max_size=int(1e6)):
        # (2) Pre-allocate numpy arrays for all components of the transition tuple.
        #     This is more memory-efficient than a Python list of dicts.
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states     = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions    = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards    = np.zeros((max_size, 1),       dtype=np.float32)
        self.next_states = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.terminals  = np.zeros((max_size, 1),       dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        """
        Store a transition at the current pointer position, then advance the pointer.
        When the buffer is full, we wrap around and overwrite the oldest transitions.
        """
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.terminals[self.ptr]   = done
        # (4) Advance the pointer and update the current buffer size.
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        """
        Pull out a mini-batch sampled uniformly at random from the stored transitions.
        Uniform sampling breaks temporal correlations between consecutive transitions.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        # (6) Convert sampled arrays to tensors and move them to the appropriate device.
        return (
            torch.tensor(self.states[idxs]),
            torch.tensor(self.actions[idxs]),
            torch.tensor(self.rewards[idxs]),
            torch.tensor(self.next_states[idxs]),
            torch.tensor(self.terminals[idxs]),
        )

    def __len__(self):
        return self.size
