"""
TD3 — Twin Delayed DDPG (Fujimoto et al., 2018).
Adapted from reinforcement_learning_2/rl/algorithms/continuous/td3.py.
"""
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from beam_optimization.algorithms.networks.policy_nets import DeterministicPolicyNetwork
from beam_optimization.algorithms.networks.value_nets   import TwinQNetwork
from beam_optimization.algorithms.utils.replay_buffer   import ReplayBuffer


class TD3:
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
                 warmup_steps: int = 1000,
                 exploration_noise: float = 0.1,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_frequency: int = 2):
        self.gamma             = gamma
        self.tau               = tau
        self.batch_size        = batch_size
        self.warmup_steps      = warmup_steps
        self.exploration_noise = exploration_noise
        self.policy_noise      = policy_noise
        self.noise_clip        = noise_clip
        self.policy_frequency  = policy_frequency
        self.update_count      = 0

        self.action_low  = torch.tensor(action_bounds[0], dtype=torch.float32)
        self.action_high = torch.tensor(action_bounds[1], dtype=torch.float32)
        # Per-dimension action half-range. exploration_noise / policy_noise /
        # noise_clip follow the TD3 paper's convention for actions normalized
        # to [-1, 1], so in this physical action space (whose per-dimension
        # bounds span several orders of magnitude) they must be rescaled by
        # each dimension's half-range to keep their intended meaning.
        self.action_halfrange = (self.action_high - self.action_low) / 2.0

        self.actor  = DeterministicPolicyNetwork(obs_dim, act_dim, action_bounds, hidden_dims)
        self.critic = TwinQNetwork(obs_dim, act_dim, hidden_dims)
        self.target_actor  = copy.deepcopy(self.actor);  [p.requires_grad_(False) for p in self.target_actor.parameters()]
        self.target_critic = copy.deepcopy(self.critic); [p.requires_grad_(False) for p in self.target_critic.parameters()]

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay = ReplayBuffer(obs_dim, act_dim, buffer_size)

    def select_action(self, state, training: bool = True):
        action = self.actor.select_action(state)
        if training:
            scale  = self.exploration_noise * self.action_halfrange.numpy()
            noise  = np.random.normal(0, scale, size=action.shape)
            action = (action + noise).clip(self.action_low.numpy(), self.action_high.numpy())
        return action

    def store(self, s, a, r, ns, done):
        self.replay.store(s, a, r, ns, float(done))

    def optimize(self):
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None
        s, a, r, ns, d = self.replay.sample(self.batch_size)

        with torch.no_grad():
            clip  = self.noise_clip * self.action_halfrange
            noise = (torch.randn_like(a) * self.policy_noise * self.action_halfrange).clamp(-clip, clip)
            na    = (self.target_actor.forward(ns) + noise).clamp(self.action_low, self.action_high)
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
            actor_loss = -self.critic.Q1(s, self.actor.forward(s)).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            for tp, sp in zip(self.target_actor.parameters(), self.actor.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            al = actor_loss.item()

        return cl.item(), al, None

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "ta": self.target_actor.state_dict(), "tc": self.target_critic.state_dict(),
            "ao": self.actor_opt.state_dict(), "co": self.critic_opt.state_dict(),
            "steps": self.update_count,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ck["actor"]); self.critic.load_state_dict(ck["critic"])
        self.target_actor.load_state_dict(ck["ta"]); self.target_critic.load_state_dict(ck["tc"])
        self.actor_opt.load_state_dict(ck["ao"]); self.critic_opt.load_state_dict(ck["co"])
        self.update_count = ck["steps"]
