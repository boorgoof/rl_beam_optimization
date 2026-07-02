"""Policy networks for continuous action spaces.
Adapted from reinforcement_learning_2/rl/networks/continuous/policy_nets.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicPolicyNetwork(nn.Module):
    """Deterministic policy π(s) → a. Used by TD3."""
    def __init__(self, obs_dim, act_dim, action_bounds, hidden_dims=(256, 256), activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.action_min = torch.tensor(action_bounds[0], dtype=torch.float32)
        self.action_max = torch.tensor(action_bounds[1], dtype=torch.float32)
        self.rescale_fn = lambda x: (((x + 1) / 2) * (self.action_max - self.action_min) + self.action_min)
        self.input_layer   = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1))
        self.output_layer  = nn.Linear(hidden_dims[-1], act_dim)

    def _fmt(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._fmt(state)
        x = self.activation_fc(self.input_layer(x))
        for h in self.hidden_layers:
            x = self.activation_fc(h(x))
        return self.rescale_fn(torch.tanh(self.output_layer(x)))

    def select_action(self, state) -> np.ndarray:
        with torch.no_grad():
            return self.forward(state).cpu().numpy().squeeze(0)


class GaussianPolicyNetwork(nn.Module):
    """Stochastic Gaussian policy π(a|s). Used by PPO and SAC."""
    def __init__(self, obs_dim, action_bounds, hidden_dims=(256, 256), activation_fc=F.relu,
                 log_std_min=-20, log_std_max=0):
        super().__init__()
        self.activation_fc = activation_fc
        self.log_std_min   = log_std_min
        self.log_std_max   = log_std_max
        self.action_min = torch.tensor(action_bounds[0], dtype=torch.float32)
        self.action_max = torch.tensor(action_bounds[1], dtype=torch.float32)
        self.rescale_fn = lambda x: (((x + 1) / 2) * (self.action_max - self.action_min) + self.action_min)
        self.target_entropy = -np.prod(self.action_max.shape)
        self.logalpha = nn.Parameter(torch.zeros(1))
        act_dim = len(self.action_max)
        self.input_layer   = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1))
        self.mean_layer    = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)

    def _fmt(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._fmt(state)
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
        action   = self.rescale_fn(tanh_a)
        log_prob = dist.log_prob(pre_tanh) - torch.log((1 - tanh_a.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, tanh_a, mean, log_std

    def select_action(self, state) -> np.ndarray:
        action, _, _, _, _ = self.full_pass(state)
        return action.detach().cpu().numpy().squeeze(0)

    def select_greedy_action(self, state) -> np.ndarray:
        mean, _ = self.forward(state)
        return self.rescale_fn(torch.tanh(mean)).detach().cpu().numpy().squeeze(0)
