"""
REINFORCE — Monte Carlo Policy Gradient (Williams, 1992).
Continuous action space with Gaussian policy.

Reference:
    Williams R.J., "Simple Statistical Gradient-Following Algorithms for
    Connectionist Reinforcement Learning", Machine Learning 8, 1992.
    https://doi.org/10.1007/BF00992696

    Sutton R.S. & Barto A.G., "Reinforcement Learning: An Introduction",
    MIT Press 2018, Ch.13. http://incompleteideas.net/book/the-book-2nd.html

    Implementation style:
    FareedKhan-dev/all-rl-algorithms
    Morales M., "Grokking Deep Reinforcement Learning", Manning 2020, Ch.11.

Gradient estimator:
    ∇J(θ) = E[Σ_t γ^t G_t ∇ log π(a_t|s_t; θ)]
    G_t = Σ_{k=t}^T γ^{k-t} r_k  (discounted return from step t)

No value baseline — high variance but conceptually the simplest policy
gradient algorithm.
"""
import numpy as np
import torch
import torch.optim as optim

from beam_optimization.algorithms.networks.policy_nets import GaussianPolicyNetwork
from beam_optimization.algorithms.utils.episode_buffer import EpisodeBuffer


class REINFORCE:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 action_bounds,
                 hidden_dims=(256, 256),
                 actor_lr: float = 7e-4,
                 gamma: float = 0.99,
                 entropy_loss_weight: float = 0.001,
                 policy_max_grad_norm: float = 1.0):
        self.gamma                = gamma
        self.entropy_loss_weight  = entropy_loss_weight
        self.policy_max_grad_norm = policy_max_grad_norm

        self.policy_network = GaussianPolicyNetwork(obs_dim, action_bounds, hidden_dims)
        self.optimizer      = optim.Adam(self.policy_network.parameters(), lr=actor_lr)
        self.episode_buffer = EpisodeBuffer(gamma=gamma, tau=1.0)

    def select_action(self, state, training: bool = True):
        if training:
            action, logpa, _, _, _ = self.policy_network.full_pass(state)
            return (action.detach().cpu().numpy().squeeze(0),
                    logpa.detach().cpu().numpy().squeeze(),
                    0.0)
        return self.policy_network.select_greedy_action(state)

    def store(self, state, action, reward, value, logpa, done):
        self.episode_buffer.store(state, action, reward, value, logpa, done)

    def optimize(self, last_value: float = 0.0):
        states, actions, returns, _, logpas = self.episode_buffer.get(last_value=0.0)
        T = len(returns)

        discounts = torch.tensor(
            np.logspace(0, T, num=T, base=self.gamma, endpoint=False),
            dtype=torch.float32)

        _, new_logpas, _, _, _ = self.policy_network.full_pass(states)
        entropy_loss = new_logpas.mean()

        policy_loss = -(discounts * returns * logpas).mean()
        loss        = policy_loss + self.entropy_loss_weight * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.policy_max_grad_norm)
        self.optimizer.step()
        return 0.0, loss.item()

    def save(self, path: str):
        torch.save({
            "policy": self.policy_network.state_dict(),
            "optim":  self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu")
        self.policy_network.load_state_dict(ck["policy"])
        self.optimizer.load_state_dict(ck["optim"])
