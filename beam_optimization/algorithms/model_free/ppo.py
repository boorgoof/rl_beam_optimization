"""
PPO — Proximal Policy Optimization (Schulman et al., 2017).
On-policy algorithm with clipped surrogate objective and GAE.
Adapted from reinforcement_learning_2/rl/algorithms/continuous/ppo.py.
"""
from typing import Optional, Union

import numpy as np
import torch
import torch.optim as optim

from beam_optimization.algorithms.networks.policy_nets import GaussianPolicyNetwork
from beam_optimization.algorithms.networks.value_nets   import ValueNetwork
from beam_optimization.algorithms.utils.episode_buffer  import EpisodeBuffer


class PPO:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 action_bounds,
                 hidden_dims=(256, 256),
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.95,
                 clip_range: float = 0.2,
                 value_clip_range: float = 0.2,
                 n_epochs: int = 10,
                 sample_ratio: float = 0.8,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 stop_kl: float = 0.02,
                 stop_mse: float = 25.0,
                 device: Optional[Union[str, torch.device]] = None):
        self.clip_range       = clip_range
        self.value_clip_range = value_clip_range
        self.n_epochs         = n_epochs
        self.sample_ratio     = sample_ratio
        self.entropy_coef     = entropy_coef
        self.max_grad_norm    = max_grad_norm
        self.stop_kl          = stop_kl
        self.stop_mse         = stop_mse
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy  = GaussianPolicyNetwork(obs_dim, action_bounds, hidden_dims).to(self.device)
        self.value   = ValueNetwork(obs_dim, hidden_dims).to(self.device)
        self.p_opt   = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.v_opt   = optim.Adam(self.value.parameters(),  lr=critic_lr)
        self.buffer  = EpisodeBuffer(gamma=gamma, tau=tau, device=self.device)

    def select_action(self, state, training: bool = True):
        if training:
            a, lp, _, _, _ = self.policy.full_pass(state)
            v = self.value(state)
            return (a.detach().cpu().numpy().squeeze(0),
                    lp.detach().cpu().numpy().squeeze(),
                    v.detach().cpu().numpy().squeeze())
        return self.policy.select_greedy_action(state)

    def store(self, state, action, reward, value, logpa, done):
        self.buffer.store(state, action, reward, value, logpa, done)

    def optimize(self, last_value: float = 0.0):
        states, actions, returns, gaes, logpas = self.buffer.get(last_value)
        n      = len(actions)
        values = self.value(states).detach()
        gaes   = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        for _ in range(self.n_epochs):
            bs  = max(1, int(self.sample_ratio * n))
            idx = np.random.choice(n, bs, replace=False)
            sb, ab, gb, lb = states[idx], actions[idx], gaes[idx], logpas[idx]

            plp = self.policy.log_prob(sb, ab)
            ratios = (plp - lb.unsqueeze(1)).exp()
            pi_obj = gb.unsqueeze(1) * ratios
            pi_clp = gb.unsqueeze(1) * ratios.clamp(1 - self.clip_range, 1 + self.clip_range)
            p_loss = -torch.min(pi_obj, pi_clp).mean() - self.entropy_coef * (-plp).mean()

            self.p_opt.zero_grad(); p_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.p_opt.step()

            with torch.no_grad():
                # k3 estimator (Schulman): E[(r-1) - log r] >= 0, lower variance
                # than the naive E[log pi_old - log pi_new] (which can go negative
                # and mask a large divergence).
                log_ratio = self.policy.log_prob(states, actions) - logpas.unsqueeze(1)
                kl = (log_ratio.exp() - 1.0 - log_ratio).mean()
            if kl.item() > self.stop_kl:
                break

        for _ in range(self.n_epochs):
            bs  = max(1, int(self.sample_ratio * n))
            idx = np.random.choice(n, bs, replace=False)
            sb, rb, vb = states[idx], returns[idx], values[idx]
            vp  = self.value(sb)
            vl  = (vp - rb).pow(2)
            vpc = vb + (vp - vb).clamp(-self.value_clip_range, self.value_clip_range)
            vlc = (vpc - rb).pow(2)
            v_loss = torch.max(vl, vlc).mul(0.5).mean()

            self.v_opt.zero_grad(); v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.v_opt.step()

            with torch.no_grad():
                mse = (values - self.value(states)).pow(2).mul(0.5).mean()
            if mse.item() > self.stop_mse:
                break

        return v_loss.item(), p_loss.item()

    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(), "value": self.value.state_dict(),
                    "po": self.p_opt.state_dict(), "vo": self.v_opt.state_dict()}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ck["policy"]); self.value.load_state_dict(ck["value"])
        self.p_opt.load_state_dict(ck["po"]);      self.v_opt.load_state_dict(ck["vo"])
