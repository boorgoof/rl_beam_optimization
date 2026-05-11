import numpy as np


class NormalNoiseDecayStrategy:
    """
    NormalNoiseDecayStrategy adds decaying Gaussian noise to the deterministic policy output.
    This is the exploration strategy used by DDPG (Morales, Ch.12).
    Unlike epsilon-greedy, here the noise is added directly to the continuous action,
    and the noise scale decays over training to reduce exploration as the policy improves.
    """
    def __init__(self, action_bounds, init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=100_000):
        # (2) Store action bounds so we can scale the noise relative to the action range.
        self.low  = np.array(action_bounds[0], dtype=np.float32)
        self.high = np.array(action_bounds[1], dtype=np.float32)
        # (3) Noise ratio starts at init_noise_ratio and decays linearly to min_noise_ratio.
        self.noise_ratio     = init_noise_ratio
        self.init_noise_ratio = init_noise_ratio
        self.min_noise_ratio  = min_noise_ratio
        self.decay_steps      = decay_steps
        self.step_count       = 0

    def select_action(self, model, state, max_exploration=False):
        """
        In max_exploration mode (e.g. for the first few episodes), use maximum noise scale.
        Otherwise adds Gaussian noise scaled by the current noise ratio to the deterministic action.
        Clips the result to the valid environment action range.
        """
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.noise_ratio * self.high
        # (5) Sample Gaussian noise centered at 0 with the current noise scale.
        noise  = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        # (6) Add the noise to the deterministic action from the policy model.
        action = model.select_action(state)
        noisy_action = action + noise
        # (7) Clip the noisy action to the valid environment action range.
        return np.clip(noisy_action, self.low, self.high)

    def _decay_noise(self):
        """
        Linearly decay the noise ratio from init_noise_ratio to min_noise_ratio
        over the specified number of decay steps.
        """
        self.step_count += 1
        fraction = min(self.step_count / self.decay_steps, 1.0)
        self.noise_ratio = self.init_noise_ratio - fraction * (self.init_noise_ratio - self.min_noise_ratio)

    def update(self):
        """Call this once per environment step to advance the noise decay schedule."""
        self._decay_noise()
