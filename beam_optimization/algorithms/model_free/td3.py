"""
TD3 — Twin Delayed DDPG (Fujimoto et al., 2018).
Adapted from reinforcement_learning_2/rl/algorithms/continuous/td3.py.
"""
import copy
from typing import Optional, Union

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from beam_optimization.algorithms.networks.policy_nets import DeterministicPolicyNetwork
from beam_optimization.algorithms.networks.value_nets   import TwinQNetwork
from beam_optimization.algorithms.utils.atomic_save     import atomic_torch_save
from beam_optimization.algorithms.utils.replay_buffer   import ReplayBuffer


class TD3:
    IMPLEMENTATION_VERSION = 2

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 action_bounds,
                 hidden_dims=(256, 256),
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 batch_size: int = 256,
                 buffer_size: int = int(1e6),
                 warmup_steps: int = 100,
                 exploration_noise: float = 0.1,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_frequency: int = 2,
                 device: Optional[Union[str, torch.device]] = None):
        self.gamma             = gamma
        self.tau               = tau
        self.batch_size        = batch_size
        self.warmup_steps      = warmup_steps
        self.exploration_noise = exploration_noise
        self.policy_noise      = policy_noise
        self.noise_clip        = noise_clip
        self.policy_frequency  = policy_frequency
        self.update_count      = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Kept as numpy too: select_action()'s exploration-noise clipping runs
        # in plain numpy every env step, where a GPU round-trip would only add
        # overhead for a batch-of-1 op.
        self._action_low_np  = np.asarray(action_bounds[0], dtype=np.float32)
        self._action_high_np = np.asarray(action_bounds[1], dtype=np.float32)

        if np.any(self._action_high_np <= self._action_low_np):
            raise ValueError("Every action upper bound must be greater than its lower bound")

        self.actor  = DeterministicPolicyNetwork(obs_dim, act_dim, action_bounds, hidden_dims).to(self.device)
        self.critic = TwinQNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
        self.target_actor  = copy.deepcopy(self.actor);  [p.requires_grad_(False) for p in self.target_actor.parameters()]
        self.target_critic = copy.deepcopy(self.critic); [p.requires_grad_(False) for p in self.target_critic.parameters()]

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay = ReplayBuffer(obs_dim, act_dim, buffer_size, device=self.device)

    def scale_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        scaled = 2.0 * (action - self._action_low_np) / (
            self._action_high_np - self._action_low_np
        ) - 1.0
        return np.clip(scaled, -1.0, 1.0)

    def unscale_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        return self._action_low_np + 0.5 * (action + 1.0) * (
            self._action_high_np - self._action_low_np
        )

    def select_action(self, state, training: bool = True):
        if training and len(self.replay) < self.warmup_steps:
            return np.random.uniform(
                self._action_low_np, self._action_high_np
            ).astype(np.float32)
        normalized_action = self.actor.select_normalized_action(state)
        if training:
            noise = np.random.normal(
                0.0, self.exploration_noise, size=normalized_action.shape
            )
            normalized_action = np.clip(normalized_action + noise, -1.0, 1.0)
        return self.unscale_action(normalized_action)

    def store(self, s, a, r, ns, done):
        self.replay.store(s, self.scale_action(a), r, ns, float(done))

    def optimize(self):
        if len(self.replay) < self.warmup_steps:
            return None
        s, a, r, ns, d = self.replay.sample(self.batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            na = (self.target_actor.normalized_forward(ns) + noise).clamp(-1.0, 1.0)
            q1t, q2t = self.target_critic(ns, na)
            tq = r + self.gamma * torch.min(q1t, q2t) * (1 - d)

        q1, q2 = self.critic(s, a)
        cl = F.mse_loss(q1, tq) + F.mse_loss(q2, tq)
        self.critic_opt.zero_grad(); cl.backward(); self.critic_opt.step()

        # Counting critic updates (not env steps) keeps the delayed policy
        # update working regardless of who fills the replay buffer (e.g. MBPO).
        self.update_count += 1
        al = 0.0
        if self.update_count % self.policy_frequency == 0:
            actor_loss = -self.critic.Q1(
                s, self.actor.normalized_forward(s)
            ).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            for tp, sp in zip(self.target_actor.parameters(), self.actor.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            al = actor_loss.item()

        return cl.item(), al, None

    def save(self, path: str, include_replay: bool = False):
        checkpoint = {
            "implementation_version": self.IMPLEMENTATION_VERSION,
            "action_representation": "normalized",
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "ta": self.target_actor.state_dict(), "tc": self.target_critic.state_dict(),
            "ao": self.actor_opt.state_dict(), "co": self.critic_opt.state_dict(),
            "steps": self.update_count,
        }
        if include_replay:
            checkpoint["replay"] = self.replay.state_dict()
        atomic_torch_save(checkpoint, path)

    def load(self, path: str, resume_training: bool = False):
        ck = torch.load(path, map_location="cpu")
        version = int(ck.get("implementation_version", 1))
        if resume_training and version != self.IMPLEMENTATION_VERSION:
            raise ValueError(
                "Legacy TD3 checkpoints use physical critic actions and cannot "
                "resume training with the normalized-action implementation. "
                "They remain valid for deterministic policy evaluation."
            )
        if resume_training and "replay" not in ck:
            raise ValueError(
                "This TD3 checkpoint has no replay snapshot. Save with "
                "include_replay=True before using resume_training=True."
            )
        self.actor.load_state_dict(ck["actor"]); self.critic.load_state_dict(ck["critic"])
        self.target_actor.load_state_dict(ck["ta"]); self.target_critic.load_state_dict(ck["tc"])
        self.actor_opt.load_state_dict(ck["ao"]); self.critic_opt.load_state_dict(ck["co"])
        self.update_count = ck["steps"]
        if resume_training:
            self.replay.load_state_dict(ck["replay"])
        self.loaded_checkpoint_version = version
