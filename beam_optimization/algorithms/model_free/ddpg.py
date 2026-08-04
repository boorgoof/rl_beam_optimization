"""
DDPG — Deep Deterministic Policy Gradient (Lillicrap et al., 2016).
Off-policy actor-critic for continuous action spaces.

Reference:
    Lillicrap T. et al., "Continuous control with deep reinforcement
    learning", ICLR 2016. https://arxiv.org/abs/1509.02971

    Original implementation style:
    FareedKhan-dev/all-rl-algorithms (08_ddpg.ipynb)
    Morales M., "Grokking Deep Reinforcement Learning", Manning 2020, Ch.9.

Key components:
    - Deterministic policy μ(s; θ)  — actor
    - Q-function Q(s, a; φ)         — critic
    - Soft target networks (Polyak averaging, τ)
    - Decaying Gaussian exploration noise
"""
import copy
from typing import Optional, Union

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from beam_optimization.algorithms.networks.policy_nets import DeterministicPolicyNetwork
from beam_optimization.algorithms.networks.value_nets  import QNetwork
from beam_optimization.algorithms.utils.atomic_save    import atomic_torch_save
from beam_optimization.algorithms.utils.replay_buffer  import ReplayBuffer
from beam_optimization.algorithms.utils.noise          import NormalNoiseDecayStrategy


class DDPG:
    IMPLEMENTATION_VERSION = 2

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 action_bounds,
                 hidden_dims=(256, 256),
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 1e-3,
                 batch_size: int = 128,
                 buffer_size: int = int(1e6),
                 warmup_steps: int = 100,
                 init_noise_ratio: float = 0.5,
                 min_noise_ratio: float = 0.01,
                 decay_steps: int = 50_000,
                 device: Optional[Union[str, torch.device]] = None):
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.warmup_steps = warmup_steps
        self.total_steps  = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.action_low = np.asarray(action_bounds[0], dtype=np.float32)
        self.action_high = np.asarray(action_bounds[1], dtype=np.float32)
        if np.any(self.action_high <= self.action_low):
            raise ValueError("Every action upper bound must be greater than its lower bound")

        self.actor  = DeterministicPolicyNetwork(obs_dim, act_dim, action_bounds, hidden_dims).to(self.device)
        self.critic = QNetwork(obs_dim, act_dim, hidden_dims).to(self.device)

        self.target_actor  = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_actor.parameters():  p.requires_grad_(False)
        for p in self.target_critic.parameters(): p.requires_grad_(False)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = ReplayBuffer(obs_dim, act_dim, buffer_size, device=self.device)
        # Pure-numpy exploration noise: model.select_action() already returns
        # a CPU numpy array regardless of the network's device (see
        # DeterministicPolicyNetwork.select_action), so no device handling
        # is needed here.
        normalized_bounds = (
            -np.ones(act_dim, dtype=np.float32),
            np.ones(act_dim, dtype=np.float32),
        )
        self.noise = NormalNoiseDecayStrategy(
            normalized_bounds, init_noise_ratio, min_noise_ratio, decay_steps
        )

    def scale_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        scaled = 2.0 * (action - self.action_low) / (
            self.action_high - self.action_low
        ) - 1.0
        return np.clip(scaled, -1.0, 1.0)

    def unscale_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        return self.action_low + 0.5 * (action + 1.0) * (
            self.action_high - self.action_low
        )

    def select_action(self, state, training: bool = True):
        if training and len(self.replay) < self.warmup_steps:
            return np.random.uniform(
                self.action_low, self.action_high
            ).astype(np.float32)
        normalized_action = self.actor.select_normalized_action(state)
        if training:
            normalized_action = self.noise.add_noise(normalized_action)
            self.noise.update()
        return self.unscale_action(normalized_action)

    def store(self, state, action, reward, next_state, done):
        self.replay.store(
            state,
            self.scale_action(action),
            reward,
            next_state,
            float(done),
        )
        self.total_steps += 1

    def optimize(self):
        if len(self.replay) < self.warmup_steps:
            return None

        s, a, r, ns, d = self.replay.sample(self.batch_size)

        with torch.no_grad():
            na      = self.target_actor.normalized_forward(ns)
            target_q = r + self.gamma * self.target_critic(ns, na) * (1 - d)
        critic_loss = F.mse_loss(self.critic(s, a), target_q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor.normalized_forward(s)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        self._soft_update(self.target_actor,  self.actor)
        self._soft_update(self.target_critic, self.critic)
        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save(self, path: str, include_replay: bool = False):
        checkpoint = {
            "implementation_version": self.IMPLEMENTATION_VERSION,
            "action_representation": "normalized",
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_steps": self.total_steps,
            "noise": self.noise.state_dict(),
        }
        if include_replay:
            checkpoint["replay"] = self.replay.state_dict()
        atomic_torch_save(checkpoint, path)

    def load(self, path: str, resume_training: bool = False):
        ck = torch.load(path, map_location="cpu")
        version = int(ck.get("implementation_version", 1))
        if resume_training and version != self.IMPLEMENTATION_VERSION:
            raise ValueError(
                "Legacy DDPG checkpoints use physical critic actions and cannot "
                "resume training with the normalized-action implementation. "
                "They remain valid for deterministic policy evaluation."
            )
        if resume_training and "replay" not in ck:
            raise ValueError(
                "This DDPG checkpoint has no replay snapshot. Save with "
                "include_replay=True before using resume_training=True."
            )
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.target_actor.load_state_dict(ck["target_actor"])
        self.target_critic.load_state_dict(ck["target_critic"])
        self.actor_opt.load_state_dict(ck["actor_opt"])
        self.critic_opt.load_state_dict(ck["critic_opt"])
        self.total_steps = ck["total_steps"]
        if "noise" in ck:
            self.noise.load_state_dict(ck["noise"])
        if resume_training:
            self.replay.load_state_dict(ck["replay"])
        self.loaded_checkpoint_version = version
