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
        noise_scale = self.high if max_exploration else self.noise_ratio * self.high
        noise  = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        return np.clip(model.select_action(state) + noise, self.low, self.high)

    def update(self):
        self.step_count += 1
        fraction = min(self.step_count / self.decay_steps, 1.0)
        self.noise_ratio = (self.init_noise_ratio
                            - fraction * (self.init_noise_ratio - self.min_noise_ratio))
