"""
SAC — Soft Actor-Critic (Haarnoja et al., 2018).
Off-policy maximum-entropy algorithm with automatic entropy tuning.
Adapted from reinforcement_learning_2/rl/algorithms/continuous/sac.py.
"""
import copy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from beam_optimization.algorithms.networks.policy_nets import GaussianPolicyNetwork
from beam_optimization.algorithms.networks.value_nets   import QNetwork
from beam_optimization.algorithms.utils.atomic_save     import atomic_torch_save
from beam_optimization.algorithms.utils.replay_buffer   import ReplayBuffer


class SAC:
    IMPLEMENTATION_VERSION = 2

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
                 warmup_steps: int = 100,
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
        # Match SB3: entropy is measured in the normalized [-1, 1] action
        # coordinates, independently of the physical units used by the env.
        self.target_entropy = -float(act_dim)

        self.action_low = np.asarray(action_bounds[0], dtype=np.float32)
        self.action_high = np.asarray(action_bounds[1], dtype=np.float32)
        if self.action_low.shape != (act_dim,) or self.action_high.shape != (act_dim,):
            raise ValueError(
                f"Expected {act_dim} action bounds, got "
                f"{self.action_low.shape} and {self.action_high.shape}"
            )
        if np.any(self.action_high <= self.action_low):
            raise ValueError("Every action upper bound must be greater than its lower bound")

        self.actor_opt  = optim.Adam(self.policy.parameters(),  lr=actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.alpha_opt   = optim.Adam([self.logalpha],           lr=alpha_lr)
        self.replay = ReplayBuffer(obs_dim, act_dim, buffer_size, device=self.device)

    def scale_action(self, action):
        """Map physical env actions to the normalized critic space [-1, 1]."""
        if isinstance(action, torch.Tensor):
            low = torch.as_tensor(self.action_low, device=action.device, dtype=action.dtype)
            high = torch.as_tensor(self.action_high, device=action.device, dtype=action.dtype)
            return torch.clamp(2.0 * (action - low) / (high - low) - 1.0, -1.0, 1.0)
        action_np = np.asarray(action, dtype=np.float32)
        scaled = 2.0 * (action_np - self.action_low) / (
            self.action_high - self.action_low
        ) - 1.0
        return np.clip(scaled, -1.0, 1.0)

    def unscale_action(self, action):
        """Map normalized critic actions back to physical env coordinates."""
        if isinstance(action, torch.Tensor):
            low = torch.as_tensor(self.action_low, device=action.device, dtype=action.dtype)
            high = torch.as_tensor(self.action_high, device=action.device, dtype=action.dtype)
            return low + 0.5 * (action + 1.0) * (high - low)
        action_np = np.asarray(action, dtype=np.float32)
        return self.action_low + 0.5 * (action_np + 1.0) * (
            self.action_high - self.action_low
        )

    def select_action(self, state, training: bool = True):
        if training and len(self.replay) < self.warmup_steps:
            return np.random.uniform(self.action_low, self.action_high).astype(np.float32)
        return (self.policy.select_action(state) if training
                else self.policy.select_greedy_action(state))

    def store(self, s, a, r, ns, done):
        self.replay.store(s, self.scale_action(a), r, ns, float(done))

    def _entropy_coefficient_loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        """SB3-compatible objective for automatic entropy tuning."""
        return -(
            self.logalpha * (log_prob + self.target_entropy).detach()
        ).mean()

    def optimize(self):
        # Like SB3, learning_starts is independent of batch size. ReplayBuffer
        # samples with replacement, so a batch of 256 is valid after 100 steps.
        if len(self.replay) < self.warmup_steps:
            return None
        s, a, r, ns, d = self.replay.sample(self.batch_size)
        alpha = self.logalpha.exp().detach()

        # Current-policy actions/log-probabilities in normalized coordinates.
        _, logpa, tanh_a, _, _ = self.policy.full_pass(
            s, include_action_scale_jacobian=False
        )

        # Match SB3's automatic entropy-coefficient update exactly: optimize
        # log(alpha), while the actor/critic use alpha detached before this step.
        ent_loss = self._entropy_coefficient_loss(logpa)
        self.alpha_opt.zero_grad()
        ent_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            _, next_logpa, next_tanh_a, _, _ = self.policy.full_pass(
                ns, include_action_scale_jacobian=False
            )
            next_q = torch.min(
                self.tc1(ns, next_tanh_a),
                self.tc2(ns, next_tanh_a),
            )
            tq = r + self.gamma * (next_q - alpha * next_logpa) * (1 - d)

        cl1 = F.mse_loss(self.critic1(s, a), tq)
        cl2 = F.mse_loss(self.critic2(s, a), tq)
        critic_loss = 0.5 * (cl1 + cl2)
        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        al = (
            alpha * logpa
            - torch.min(self.critic1(s, tanh_a), self.critic2(s, tanh_a))
        ).mean()
        self.actor_opt.zero_grad(); al.backward(); self.actor_opt.step()

        for tp, sp in zip(self.tc1.parameters(), self.critic1.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        for tp, sp in zip(self.tc2.parameters(), self.critic2.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), al.item(), ent_loss.item()

    def save(self, path: str, include_replay: bool = False):
        checkpoint = {
            "implementation_version": self.IMPLEMENTATION_VERSION,
            "action_representation": "normalized",
            "policy": self.policy.state_dict(),
            "c1": self.critic1.state_dict(), "c2": self.critic2.state_dict(),
            "tc1": self.tc1.state_dict(),    "tc2": self.tc2.state_dict(),
            "logalpha": self.logalpha.detach(),
            "a_opt": self.actor_opt.state_dict(),
            "c1_opt": self.critic1_opt.state_dict(), "c2_opt": self.critic2_opt.state_dict(),
            "al_opt": self.alpha_opt.state_dict(),
        }
        if include_replay:
            checkpoint["replay"] = self.replay.state_dict()
        atomic_torch_save(checkpoint, path)

    def load(self, path: str, resume_training: bool = False):
        ck = torch.load(path, map_location="cpu")
        version = int(ck.get("implementation_version", 1))
        if resume_training and version != self.IMPLEMENTATION_VERSION:
            raise ValueError(
                "Legacy SAC checkpoints use physical critic actions and cannot "
                "resume training with the normalized-action implementation. "
                "They remain valid for deterministic policy evaluation."
            )
        if resume_training and "replay" not in ck:
            raise ValueError(
                "This SAC checkpoint has no replay snapshot. Save with "
                "include_replay=True before using resume_training=True."
            )
        self.policy.load_state_dict(ck["policy"])
        self.critic1.load_state_dict(ck["c1"]); self.critic2.load_state_dict(ck["c2"])
        self.tc1.load_state_dict(ck["tc1"]);    self.tc2.load_state_dict(ck["tc2"])
        self.logalpha.data.copy_(ck["logalpha"])
        self.actor_opt.load_state_dict(ck["a_opt"])
        self.critic1_opt.load_state_dict(ck["c1_opt"]); self.critic2_opt.load_state_dict(ck["c2_opt"])
        self.alpha_opt.load_state_dict(ck["al_opt"])
        if resume_training:
            self.replay.load_state_dict(ck["replay"])
        self.loaded_checkpoint_version = version
