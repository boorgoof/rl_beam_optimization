"""
SAC — Soft Actor-Critic (Haarnoja et al., 2018).
Off-policy maximum-entropy algorithm with automatic entropy tuning.
Adapted from reinforcement_learning_2/rl/algorithms/continuous/sac.py.
"""
import copy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from beam_optimization.algorithms.networks.policy_nets import GaussianPolicyNetwork
from beam_optimization.algorithms.networks.value_nets   import QNetwork
from beam_optimization.algorithms.utils.replay_buffer   import ReplayBuffer


class SAC:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 action_bounds,
                 hidden_dims=(256, 256),
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 batch_size: int = 256,
                 buffer_size: int = int(1e6),
                 warmup_steps: int = 1000,
                 device: Optional[Union[str, torch.device]] = None):
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.warmup_steps = warmup_steps
        # SAC is a plain class, not an nn.Module: each sub-network is moved to
        # `device` individually (module.to() does not cascade automatically
        # across unrelated attributes the way it does for registered children).
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy  = GaussianPolicyNetwork(obs_dim, action_bounds, hidden_dims).to(self.device)
        self.critic1 = QNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
        self.critic2 = QNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
        self.tc1 = copy.deepcopy(self.critic1); [p.requires_grad_(False) for p in self.tc1.parameters()]
        self.tc2 = copy.deepcopy(self.critic2); [p.requires_grad_(False) for p in self.tc2.parameters()]

        self.logalpha       = nn.Parameter(torch.zeros(1, device=self.device))
        # The standard SAC target entropy -act_dim assumes actions normalized
        # to [-1, 1]. In this physical action space each dimension i has range
        # (hi_i - lo_i), which shifts the policy's log-density by
        # -log((hi_i - lo_i)/2) per dimension; the target must shift by the
        # same log-volume or alpha auto-tuning chases an unreachable entropy
        # and collapses to zero.
        low  = np.asarray(action_bounds[0], dtype=np.float64)
        high = np.asarray(action_bounds[1], dtype=np.float64)
        self.target_entropy = float(-act_dim + np.sum(np.log((high - low) / 2.0)))

        self.actor_opt  = optim.Adam(self.policy.parameters(),  lr=actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.alpha_opt   = optim.Adam([self.logalpha],           lr=alpha_lr)
        self.replay = ReplayBuffer(obs_dim, act_dim, buffer_size, device=self.device)

    def select_action(self, state, training: bool = True):
        return (self.policy.select_action(state) if training
                else self.policy.select_greedy_action(state))

    def store(self, s, a, r, ns, done):
        self.replay.store(s, a, r, ns, float(done))

    def optimize(self):
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None
        s, a, r, ns, d = self.replay.sample(self.batch_size)
        alpha = self.logalpha.exp().detach()

        with torch.no_grad():
            na, nlp, _, _, _ = self.policy.full_pass(ns)
            tq = r + self.gamma * (torch.min(self.tc1(ns, na), self.tc2(ns, na)) - alpha * nlp) * (1 - d)

        cl1 = F.mse_loss(self.critic1(s, a), tq)
        cl2 = F.mse_loss(self.critic2(s, a), tq)
        self.critic1_opt.zero_grad(); cl1.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); cl2.backward(); self.critic2_opt.step()

        na, nlp, _, _, _ = self.policy.full_pass(s)
        al = (alpha * nlp - torch.min(self.critic1(s, na), self.critic2(s, na))).mean()
        self.actor_opt.zero_grad(); al.backward(); self.actor_opt.step()

        ent_loss = -(self.logalpha.exp() * (nlp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); ent_loss.backward(); self.alpha_opt.step()

        for tp, sp in zip(self.tc1.parameters(), self.critic1.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        for tp, sp in zip(self.tc2.parameters(), self.critic2.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return (cl1.item() + cl2.item()) / 2, al.item(), ent_loss.item()

    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "c1": self.critic1.state_dict(), "c2": self.critic2.state_dict(),
            "tc1": self.tc1.state_dict(),    "tc2": self.tc2.state_dict(),
            "logalpha": self.logalpha.detach(),
            "a_opt": self.actor_opt.state_dict(),
            "c1_opt": self.critic1_opt.state_dict(), "c2_opt": self.critic2_opt.state_dict(),
            "al_opt": self.alpha_opt.state_dict(),
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ck["policy"])
        self.critic1.load_state_dict(ck["c1"]); self.critic2.load_state_dict(ck["c2"])
        self.tc1.load_state_dict(ck["tc1"]);    self.tc2.load_state_dict(ck["tc2"])
        self.logalpha.data.copy_(ck["logalpha"])
        self.actor_opt.load_state_dict(ck["a_opt"])
        self.critic1_opt.load_state_dict(ck["c1_opt"]); self.critic2_opt.load_state_dict(ck["c2_opt"])
        self.alpha_opt.load_state_dict(ck["al_opt"])
