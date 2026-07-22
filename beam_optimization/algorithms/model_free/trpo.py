"""
TRPO — Trust Region Policy Optimization (Schulman et al., 2015).
Continuous action space with Gaussian policy.

Reference:
    Schulman J. et al., "Trust Region Policy Optimization", ICML 2015.
    https://arxiv.org/abs/1502.05477

    Implementation style:
    FareedKhan-dev/all-rl-algorithms (12_trpo.ipynb)
    Morales M., "Grokking Deep Reinforcement Learning", Manning 2020, Ch.12.

Core idea:
    Instead of clipping the probability ratio (PPO), TRPO enforces a hard
    KL constraint via a trust region:

        max_θ  E[r_t(θ) * A_t]
        s.t.   KL(π_old || π_new) ≤ δ

    Solved with:
        1. Policy gradient g = ∇_θ L(θ)
        2. Natural gradient: d = F^{-1} g  via conjugate gradient
        3. Step size:        β = √(2δ / d^T F d)
        4. Backtracking line search to enforce KL ≤ δ
        5. Critic updated separately with Adam (MSE on returns)
"""
from typing import Optional, Union

import torch
import torch.optim as optim

from beam_optimization.algorithms.networks.policy_nets  import GaussianPolicyNetwork
from beam_optimization.algorithms.networks.value_nets   import ValueNetwork
from beam_optimization.algorithms.utils.episode_buffer  import EpisodeBuffer
from beam_optimization.algorithms.utils.trpo_utils import (
    conjugate_gradient, fisher_vector_product, compute_kl_divergence,
    get_flat_params, set_flat_params, get_flat_grad, line_search,
)


class TRPO:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 action_bounds,
                 hidden_dims=(256, 256),
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.97,
                 max_kl: float = 0.01,
                 cg_steps: int = 10,
                 cg_damping: float = 0.1,
                 value_epochs: int = 5,
                 device: Optional[Union[str, torch.device]] = None):
        self.max_kl       = max_kl
        self.cg_steps     = cg_steps
        self.cg_damping   = cg_damping
        self.value_epochs = value_epochs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_network = GaussianPolicyNetwork(obs_dim, action_bounds, hidden_dims).to(self.device)
        self.value_network  = ValueNetwork(obs_dim, hidden_dims).to(self.device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=critic_lr)
        self.episode_buffer  = EpisodeBuffer(gamma=gamma, tau=tau, device=self.device)

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
        states, actions, returns, gaes, old_logpas = self.episode_buffer.get(last_value)
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        with torch.no_grad():
            mean_old, log_std_old = self.policy_network.forward(states)
            mean_old    = mean_old.detach()
            log_std_old = log_std_old.detach()
        old_dist_params = (mean_old, log_std_old)

        log_probs   = self.policy_network.log_prob(states, actions)
        surr_loss   = -(gaes.unsqueeze(1) * (log_probs - old_logpas.unsqueeze(1)).exp()).mean()
        policy_grad = get_flat_grad(surr_loss, self.policy_network, retain_graph=True)

        def Avp(v):
            return fisher_vector_product(
                self.policy_network, states, v, self.cg_damping)

        natural_grad = conjugate_gradient(Avp, -policy_grad, n_steps=self.cg_steps)
        dFd       = (natural_grad * Avp(natural_grad)).sum()
        step_size = (2 * self.max_kl / (dFd + 1e-8)).sqrt()
        full_step = step_size * natural_grad

        line_search(
            policy=self.policy_network,
            states=states, actions=actions, advantages=gaes,
            old_log_probs=old_logpas.unsqueeze(1),
            old_dist_params=old_dist_params,
            full_step=full_step, max_kl=self.max_kl,
        )

        value_loss_val = 0.0
        for _ in range(self.value_epochs):
            values_pred = self.value_network(states)
            value_loss  = (values_pred - returns.unsqueeze(1)).pow(2).mul(0.5).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            value_loss_val = value_loss.item()

        return value_loss_val, surr_loss.item()

    def save(self, path: str):
        torch.save({
            "policy": self.policy_network.state_dict(),
            "value":  self.value_network.state_dict(),
            "v_opt":  self.value_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu")
        self.policy_network.load_state_dict(ck["policy"])
        self.value_network.load_state_dict(ck["value"])
        self.value_optimizer.load_state_dict(ck["v_opt"])
