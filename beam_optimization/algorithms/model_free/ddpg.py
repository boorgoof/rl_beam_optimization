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

import torch
import torch.optim as optim
import torch.nn.functional as F

from beam_optimization.algorithms.networks.policy_nets import DeterministicPolicyNetwork
from beam_optimization.algorithms.networks.value_nets  import QNetwork
from beam_optimization.algorithms.utils.replay_buffer  import ReplayBuffer
from beam_optimization.algorithms.utils.noise          import NormalNoiseDecayStrategy


class DDPG:
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
                 warmup_steps: int = 1000,
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
        self.noise  = NormalNoiseDecayStrategy(
            action_bounds, init_noise_ratio, min_noise_ratio, decay_steps)

    def select_action(self, state, training: bool = True):
        if training:
            action = self.noise.select_action(self.actor, state)
            self.noise.update()
            return action
        return self.actor.select_action(state)

    def store(self, state, action, reward, next_state, done):
        self.replay.store(state, action, reward, next_state, float(done))
        self.total_steps += 1

    def optimize(self):
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None

        s, a, r, ns, d = self.replay.sample(self.batch_size)

        with torch.no_grad():
            na      = self.target_actor.forward(ns)
            target_q = r + self.gamma * self.target_critic(ns, na) * (1 - d)
        critic_loss = F.mse_loss(self.critic(s, a), target_q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor.forward(s)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        self._soft_update(self.target_actor,  self.actor)
        self._soft_update(self.target_critic, self.critic)
        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.target_actor.load_state_dict(ck["target_actor"])
        self.target_critic.load_state_dict(ck["target_critic"])
        self.actor_opt.load_state_dict(ck["actor_opt"])
        self.critic_opt.load_state_dict(ck["critic_opt"])
        self.total_steps = ck["total_steps"]
