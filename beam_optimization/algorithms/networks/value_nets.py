"""Value networks for continuous action spaces.
Adapted from reinforcement_learning_2/rl/networks/continuous/value_nets.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from beam_optimization.algorithms.networks.policy_nets import format_input


class ValueNetwork(nn.Module):
    """State-value V(s) → scalar. Used by A2C, PPO and TRPO."""
    def __init__(self, obs_dim, hidden_dims=(256, 256), activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.input_layer   = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1))
        self.output_layer  = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state):
        x = format_input(state)
        x = self.activation_fc(self.input_layer(x))
        for h in self.hidden_layers:
            x = self.activation_fc(h(x))
        return self.output_layer(x)


class QNetwork(nn.Module):
    """State-action value Q(s,a) → scalar. Used by DDPG and SAC."""
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.input_layer   = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1))
        self.output_layer  = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state, action):
        x = torch.cat((format_input(state), format_input(action)), dim=1)
        x = self.activation_fc(self.input_layer(x))
        for h in self.hidden_layers:
            x = self.activation_fc(h(x))
        return self.output_layer(x)


class TwinQNetwork(nn.Module):
    """Two independent Q-networks for overestimation reduction. Used by TD3."""
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), activation_fc=F.relu):
        super().__init__()
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dims, activation_fc)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dims, activation_fc)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def Q1(self, state, action):
        return self.q1(state, action)
