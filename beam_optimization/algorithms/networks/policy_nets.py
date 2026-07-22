"""Policy networks for continuous action spaces.
Adapted from reinforcement_learning_2/rl/networks/continuous/policy_nets.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def format_input(x, device=None):
    """Convert numpy/list input to a batched float32 tensor, optionally moved to `device`."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
    if device is not None:
        x = x.to(device)
    return x


class DeterministicPolicyNetwork(nn.Module):
    """Deterministic policy π(s) → a. Used by DDPG and TD3."""
    def __init__(self, obs_dim, act_dim, action_bounds, hidden_dims=(256, 256), activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.register_buffer("action_min", torch.tensor(action_bounds[0], dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(action_bounds[1], dtype=torch.float32))
        self.input_layer   = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1))
        self.output_layer  = nn.Linear(hidden_dims[-1], act_dim)

    def _rescale(self, x):
        return ((x + 1) / 2) * (self.action_max - self.action_min) + self.action_min

    def forward(self, state):
        x = format_input(state, device=self.action_min.device)
        x = self.activation_fc(self.input_layer(x))
        for h in self.hidden_layers:
            x = self.activation_fc(h(x))
        return self._rescale(torch.tanh(self.output_layer(x)))

    def select_action(self, state) -> np.ndarray:
        with torch.no_grad():
            return self.forward(state).cpu().numpy().squeeze(0)


class GaussianPolicyNetwork(nn.Module):
    """Squashed-Gaussian policy π(a|s): Normal → tanh → rescale to bounds.
    Used by REINFORCE, A2C, PPO, TRPO, SAC and SVG.
    """
    def __init__(self, obs_dim, action_bounds, hidden_dims=(256, 256), activation_fc=F.relu,
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.activation_fc = activation_fc
        self.log_std_min   = log_std_min
        self.log_std_max   = log_std_max
        self.register_buffer("action_min", torch.tensor(action_bounds[0], dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(action_bounds[1], dtype=torch.float32))
        act_dim = len(action_bounds[1])
        self.input_layer   = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1))
        self.mean_layer    = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)

    def _rescale(self, x):
        return ((x + 1) / 2) * (self.action_max - self.action_min) + self.action_min

    def _log_rescale_jacobian(self):
        # a = (tanh(u)+1)/2 * (hi-lo) + lo  ⇒  |da/d tanh(u)| = (hi-lo)/2
        return torch.log((self.action_max - self.action_min) / 2)

    def forward(self, state):
        x = format_input(state, device=self.action_min.device)
        x = self.activation_fc(self.input_layer(x))
        for h in self.hidden_layers:
            x = self.activation_fc(h(x))
        mean    = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def full_pass(self, state):
        mean, log_std = self.forward(state)
        std      = log_std.exp()
        dist     = torch.distributions.Normal(mean, std)
        pre_tanh = dist.rsample()
        tanh_a   = torch.tanh(pre_tanh)
        action   = self._rescale(tanh_a)
        log_prob = (dist.log_prob(pre_tanh)
                    - torch.log((1 - tanh_a.pow(2)) + 1e-6)
                    - self._log_rescale_jacobian())
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, tanh_a, mean, log_std

    def log_prob(self, states, actions):
        """Log-probability of already-taken (physical-space) actions under the
        current policy, with gradient. Inverts the rescale+tanh squashing."""
        mean, log_std = self.forward(states)
        dist   = torch.distributions.Normal(mean, log_std.exp())
        tanh_a = (actions - self.action_min) / (self.action_max - self.action_min) * 2 - 1
        tanh_a = tanh_a.clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh = torch.atanh(tanh_a)
        log_prob = (dist.log_prob(pre_tanh)
                    - torch.log((1 - tanh_a.pow(2)) + 1e-6)
                    - self._log_rescale_jacobian())
        return log_prob.sum(dim=-1, keepdim=True)

    def select_action(self, state) -> np.ndarray:
        action, _, _, _, _ = self.full_pass(state)
        return action.detach().cpu().numpy().squeeze(0)

    def select_greedy_action(self, state) -> np.ndarray:
        mean, _ = self.forward(state)
        return self._rescale(torch.tanh(mean)).detach().cpu().numpy().squeeze(0)
