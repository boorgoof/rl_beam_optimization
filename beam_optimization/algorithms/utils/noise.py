"""Exploration noise strategies for deterministic policies.
Adapted from reinforcement_learning_2/rl/utils/noise.py.
"""
import numpy as np


class NormalNoiseDecayStrategy:
    """Decaying Gaussian exploration noise for DDPG.

    Adds Gaussian noise to the deterministic policy output and linearly
    decays the noise scale over training.

    Reference:
        Lillicrap et al., "Continuous control with deep reinforcement
        learning", ICLR 2016. https://arxiv.org/abs/1509.02971
    """

    def __init__(self, action_bounds, init_noise_ratio=0.5,
                 min_noise_ratio=0.1, decay_steps=100_000):
        self.low  = np.array(action_bounds[0], dtype=np.float32)
        self.high = np.array(action_bounds[1], dtype=np.float32)
        self.noise_ratio      = init_noise_ratio
        self.init_noise_ratio = init_noise_ratio
        self.min_noise_ratio  = min_noise_ratio
        self.decay_steps      = decay_steps
        self.step_count       = 0

    def select_action(self, model, state, max_exploration=False):
        return self.add_noise(model.select_action(state), max_exploration)

    def add_noise(self, action, max_exploration=False):
        """Add bounded Gaussian noise to an already-computed action."""
        noise_scale = self.high if max_exploration else self.noise_ratio * self.high
        noise  = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        return np.clip(np.asarray(action) + noise, self.low, self.high)

    def update(self):
        self.step_count += 1
        fraction = min(self.step_count / self.decay_steps, 1.0)
        self.noise_ratio = (self.init_noise_ratio
                            - fraction * (self.init_noise_ratio - self.min_noise_ratio))

    def state_dict(self):
        """Return the mutable schedule state needed to resume exploration."""
        return {
            "step_count": int(self.step_count),
            "noise_ratio": float(self.noise_ratio),
        }

    def load_state_dict(self, state):
        """Restore a state produced by state_dict(), validating its range."""
        step_count = int(state["step_count"])
        noise_ratio = float(state["noise_ratio"])
        if step_count < 0:
            raise ValueError("Noise step_count must be non-negative")
        if not np.isfinite(noise_ratio):
            raise ValueError("Noise ratio must be finite")
        self.step_count = step_count
        self.noise_ratio = noise_ratio
