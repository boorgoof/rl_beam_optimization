import copy
import numpy as np
import torch
import torch.optim as optim

from rl_continuous.networks.policy_nets import FCDP
from rl_continuous.networks.value_nets import FCTQV
from rl_continuous.utils.replay_buffer import ReplayBuffer
from rl_continuous.utils.noise import NormalNoiseDecayStrategy


class TD3:
    """
    TD3 stands for Twin Delayed Deep Deterministic Policy Gradient (Fujimoto et al., 2018).
    TD3 improves DDPG with three specific fixes (Morales, Ch.12):
      1. Twin critics: two independent Q-networks; use min(Q_a, Q_b) to reduce overestimation.
      2. Target policy smoothing: add clipped noise to target actions to regularize the critic.
      3. Delayed policy updates: update the actor less frequently than the critics.
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 action_bounds,
                 hidden_dims=(256, 256),
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 buffer_size=int(1e6),
                 batch_size=256,
                 warmup_steps=1000,
                 policy_max_grad_norm=float('inf'),
                 value_max_grad_norm=float('inf'),
                 policy_noise_ratio=0.2,
                 policy_noise_clip_ratio=0.5,
                 train_policy_every_steps=2,
                 noise_init_ratio=0.5,
                 noise_min_ratio=0.1,
                 noise_decay_steps=100_000):
        # (2) Store hyperparameters.
        self.gamma                    = gamma
        self.tau                      = tau
        self.batch_size               = batch_size
        self.warmup_steps             = warmup_steps
        self.policy_max_grad_norm     = policy_max_grad_norm
        self.value_max_grad_norm      = value_max_grad_norm
        self.policy_noise_ratio       = policy_noise_ratio
        self.policy_noise_clip_ratio  = policy_noise_clip_ratio
        self.train_policy_every_steps = train_policy_every_steps
        self._policy_update_counter   = 0

        # (3) Build the online policy model mu(s; phi) — deterministic actor.
        self.online_policy_model = FCDP(obs_dim, action_bounds, hidden_dims)
        # (4) Build the target policy model mu(s; phi^-) — used for computing smoothed targets.
        self.target_policy_model = copy.deepcopy(self.online_policy_model)

        # (5) Build the online twin Q-value model Q_a, Q_b (s, a; theta) — twin critics.
        self.online_value_model = FCTQV(obs_dim, act_dim, hidden_dims)
        # (6) Build the target twin Q-value model — a slowly-updated copy for stable targets.
        self.target_value_model = copy.deepcopy(self.online_value_model)

        # (7) Target networks are never directly optimized.
        for p in self.target_policy_model.parameters():
            p.requires_grad = False
        for p in self.target_value_model.parameters():
            p.requires_grad = False

        # (8) Store action bounds as tensors for the target smoothing clamp operation.
        self.env_min = torch.tensor(action_bounds[0], dtype=torch.float32)
        self.env_max = torch.tensor(action_bounds[1], dtype=torch.float32)

        # (9) Build optimizers.
        self.policy_optimizer = optim.Adam(self.online_policy_model.parameters(), lr=actor_lr)
        self.value_optimizer  = optim.Adam(self.online_value_model.parameters(), lr=critic_lr)

        # (10) Build replay buffer and noise strategy for environment exploration.
        self.replay_buffer  = ReplayBuffer(obs_dim, act_dim, buffer_size)
        self.noise_strategy = NormalNoiseDecayStrategy(
            action_bounds, noise_init_ratio, noise_min_ratio, noise_decay_steps)

        self.total_steps = 0

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, state, training=True):
        """Add exploration noise during training; use clean policy for evaluation."""
        if training:
            max_exploration = self.total_steps < self.warmup_steps
            return self.noise_strategy.select_action(
                self.online_policy_model, state, max_exploration)
        else:
            return self.online_policy_model.select_action(state)

    def store(self, state, action, reward, next_state, done):
        """Store transition and advance the noise decay schedule."""
        self.replay_buffer.store(state, action, reward, next_state, done)
        self.noise_strategy.update()
        self.total_steps += 1

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self):
        # (13) Only start optimizing once the buffer has enough samples.
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # (14) Pull out a mini-batch of transitions from the replay buffer.
        states, actions, rewards, next_states, is_terminals = \
            self.replay_buffer.sample(self.batch_size)

        # --- Improvement 1: Twin critics + target smoothing ---

        # (15) Compute the action range for scaling noise.
        a_ran = self.env_max - self.env_min

        # (16) Sample target smoothing noise: epsilon ~ N(0, policy_noise_ratio * a_range).
        #      Target smoothing regularizes the critic by forcing it to be smooth across
        #      nearby actions, preventing it from exploiting narrow Q-function peaks.
        a_noise = torch.randn_like(actions) * self.policy_noise_ratio * a_ran

        # (17) Clip the noise to a fraction of the action range to prevent large perturbations.
        n_min = self.env_min * self.policy_noise_clip_ratio
        n_max = self.env_max * self.policy_noise_clip_ratio
        a_noise = torch.max(torch.min(a_noise, n_max), n_min)

        # (18) Get the target policy action and add the clipped noise.
        argmax_a_q_sp = self.target_policy_model(next_states)
        noisy_argmax_a_q_sp = argmax_a_q_sp + a_noise
        # (19) Clip the smoothed action back to the valid action range.
        noisy_argmax_a_q_sp = torch.max(
            torch.min(noisy_argmax_a_q_sp, self.env_max), self.env_min)

        # (20) Get Q-values from both target streams and use the MINIMUM to form the target.
        #      TD3 target: y = r + gamma * min(Q_a, Q_b)(s', a'^smooth)
        #      Using the minimum reduces overestimation bias from function approximation.
        max_a_q_sp_a, max_a_q_sp_b = self.target_value_model(next_states, noisy_argmax_a_q_sp)
        max_a_q_sp = torch.min(max_a_q_sp_a, max_a_q_sp_b)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)

        # (21) Get predictions from both online streams and compute MSE loss for each.
        q_sa_a, q_sa_b = self.online_value_model(states, actions)
        td_error_a  = q_sa_a - target_q_sa.detach()
        td_error_b  = q_sa_b - target_q_sa.detach()
        value_loss  = td_error_a.pow(2).mul(0.5).mean() + td_error_b.pow(2).mul(0.5).mean()

        # (22) Optimize the twin value network.
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_value_model.parameters(), self.value_max_grad_norm)
        self.value_optimizer.step()

        # --- Improvement 3: Delayed policy update ---

        policy_loss = None
        self._policy_update_counter += 1
        # (23) Update the policy and target networks only every train_policy_every_steps critic steps.
        #      Delaying the actor update gives the critics time to converge before shaping the policy.
        if self._policy_update_counter % self.train_policy_every_steps == 0:

            # (24) Policy loss: L(phi) = -E_s[Q_a(s, mu(s; phi); theta)]
            #      We use only stream A (Qa) for the policy gradient, as in Morales/Fujimoto.
            argmax_a_q_s = self.online_policy_model(states)
            max_a_q_s    = self.online_value_model.Qa(states, argmax_a_q_s)
            policy_loss  = -max_a_q_s.mean()

            # (25) Optimize the policy network.
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.online_policy_model.parameters(), self.policy_max_grad_norm)
            self.policy_optimizer.step()

            # (26) Soft-update both target networks after each policy update.
            self._soft_update(self.online_value_model,  self.target_value_model)
            self._soft_update(self.online_policy_model, self.target_policy_model)

            policy_loss = policy_loss.item()

        return value_loss.item(), policy_loss

    def _soft_update(self, online, target):
        """Polyak averaging: theta^- <- tau * theta + (1 - tau) * theta^-."""
        for online_p, target_p in zip(online.parameters(), target.parameters()):
            target_p.data.copy_(
                self.tau * online_p.data + (1.0 - self.tau) * target_p.data)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            'online_policy': self.online_policy_model.state_dict(),
            'target_policy': self.target_policy_model.state_dict(),
            'online_value':  self.online_value_model.state_dict(),
            'target_value':  self.target_value_model.state_dict(),
            'policy_opt':    self.policy_optimizer.state_dict(),
            'value_opt':     self.value_optimizer.state_dict(),
            'total_steps':   self.total_steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.online_policy_model.load_state_dict(ckpt['online_policy'])
        self.target_policy_model.load_state_dict(ckpt['target_policy'])
        self.online_value_model.load_state_dict(ckpt['online_value'])
        self.target_value_model.load_state_dict(ckpt['target_value'])
        self.policy_optimizer.load_state_dict(ckpt['policy_opt'])
        self.value_optimizer.load_state_dict(ckpt['value_opt'])
        self.total_steps = ckpt['total_steps']
