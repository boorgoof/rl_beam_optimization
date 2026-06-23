"""Value networks for continuous action spaces.
Adapted from reinforcement_learning_2/rl/networks/continuous/value_nets.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """State-value V(s) → scalar. Used by PPO."""
    def __init__(self, obs_dim, hidden_dims=(256, 256), activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.input_layer   = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1))
        self.output_layer  = nn.Linear(hidden_dims[-1], 1)

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
        return self.output_layer(x)


class QNetwork(nn.Module):
    """State-action value Q(s,a) → scalar. Used by SAC."""
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.input_layer   = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1))
        self.output_layer  = nn.Linear(hidden_dims[-1], 1)

    def _fmt(self, s, a):
        def _t(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
            return x
        return _t(s), _t(a)

    def forward(self, state, action):
        x, u = self._fmt(state, action)
        x = torch.cat((x, u), dim=1)
        x = self.activation_fc(self.input_layer(x))
        for h in self.hidden_layers:
            x = self.activation_fc(h(x))
        return self.output_layer(x)


class TwinQNetwork(nn.Module):
    """Twin Q-networks for overestimation reduction. Used by TD3 and SAC."""
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.input_a  = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        self.hidden_a = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1))
        self.output_a = nn.Linear(hidden_dims[-1], 1)
        self.input_b  = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        self.hidden_b = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1))
        self.output_b = nn.Linear(hidden_dims[-1], 1)

    def _fmt(self, s, a):
        def _t(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
            return x
        return _t(s), _t(a)

    def forward(self, state, action):
        x, u = self._fmt(state, action)
        xu = torch.cat((x, u), dim=1)
        xa = self.activation_fc(self.input_a(xu))
        for h in self.hidden_a:
            xa = self.activation_fc(h(xa))
        q1 = self.output_a(xa)
        xb = self.activation_fc(self.input_b(xu))
        for h in self.hidden_b:
            xb = self.activation_fc(h(xb))
        q2 = self.output_b(xb)
        return q1, q2

    def Q1(self, state, action):
        x, u = self._fmt(state, action)
        xu = torch.cat((x, u), dim=1)
        xa = self.activation_fc(self.input_a(xu))
        for h in self.hidden_a:
            xa = self.activation_fc(h(xa))
        return self.output_a(xa)
