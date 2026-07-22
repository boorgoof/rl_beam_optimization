"""
A2C — Advantage Actor-Critic, synchronous version (Mnih et al., 2016).
Continuous action space. Two separate networks with separate optimizers.

Reference:
    Mnih V. et al., "Asynchronous Methods for Deep Reinforcement Learning",
    ICML 2016. https://arxiv.org/abs/1602.01783  (A3C; A2C is synchronous variant)

    Implementation style:
    FareedKhan-dev/all-rl-algorithms (07_a2c.ipynb)
    Morales M., "Grokking Deep Reinforcement Learning", Manning 2020, Ch.11.

Algorithm:
    Collect one episode → compute GAE advantages → one gradient step each on
    actor (policy gradient + entropy bonus) and critic (MSE on returns).
    No multiple epochs — that is PPO's improvement.
"""
from typing import Optional, Union

import torch
import torch.optim as optim

from beam_optimization.algorithms.networks.policy_nets import GaussianPolicyNetwork
from beam_optimization.algorithms.networks.value_nets  import ValueNetwork
from beam_optimization.algorithms.utils.episode_buffer import EpisodeBuffer


class A2C:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 action_bounds,
                 hidden_dims=(256, 256),
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.95,
                 entropy_loss_weight: float = 0.001,
                 policy_max_grad_norm: float = 1.0,
                 device: Optional[Union[str, torch.device]] = None):
        self.entropy_loss_weight  = entropy_loss_weight
        self.policy_max_grad_norm = policy_max_grad_norm
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_network = GaussianPolicyNetwork(obs_dim, action_bounds, hidden_dims).to(self.device)
        self.value_network  = ValueNetwork(obs_dim, hidden_dims).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=actor_lr)
        self.value_optimizer  = optim.Adam(self.value_network.parameters(),  lr=critic_lr)

        self.episode_buffer = EpisodeBuffer(gamma=gamma, tau=tau, device=self.device)

    def select_action(self, state, training: bool = True):
        if training:
            action, logpa, _, _, _ = self.policy_network.full_pass(state)
            value = self.value_network(state)
            return (action.detach().cpu().numpy().squeeze(0),
                    logpa.detach().cpu().numpy().squeeze(),
                    value.detach().cpu().numpy().squeeze())
        return self.policy_network.select_greedy_action(state)

    def store(self, state, action, reward, value, logpa, done):
        self.episode_buffer.store(state, action, reward, value, logpa, done)

    def optimize(self, last_value: float = 0.0):
        states, actions, returns, gaes, logpas = self.episode_buffer.get(last_value)
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        log_prob = self.policy_network.log_prob(states, actions)
        entropy  = -log_prob

        policy_loss = -(gaes.unsqueeze(1) * log_prob).mean()
        actor_loss  = policy_loss - self.entropy_loss_weight * entropy.mean()

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.policy_max_grad_norm)
        self.policy_optimizer.step()

        values_pred = self.value_network(states)
        value_loss  = (values_pred - returns.unsqueeze(1)).pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return value_loss.item(), policy_loss.item()

    def save(self, path: str):
        torch.save({
            "policy": self.policy_network.state_dict(),
            "value":  self.value_network.state_dict(),
            "p_opt":  self.policy_optimizer.state_dict(),
            "v_opt":  self.value_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu")
        self.policy_network.load_state_dict(ck["policy"])
        self.value_network.load_state_dict(ck["value"])
        self.policy_optimizer.load_state_dict(ck["p_opt"])
        self.value_optimizer.load_state_dict(ck["v_opt"])
