import copy
import numpy as np
import torch
import torch.optim as optim

from rl_continuous.networks.policy_nets import FCDP
from rl_continuous.networks.value_nets import FCQV
from rl_continuous.utils.replay_buffer import ReplayBuffer
from rl_continuous.utils.noise import NormalNoiseDecayStrategy


class DDPG:
    """
    DDPG stands for Deep Deterministic Policy Gradient (Lillicrap et al., 2015).
    It extends DQN to continuous action spaces by learning a deterministic policy mu(s)
    that approximates the argmax of the Q-function — avoiding the expensive max over actions.
    Key components (Morales, Ch.12):
      - Deterministic policy:  mu(s; phi)
      - Q-value function:      Q(s, a; theta)
      - Replay buffer for off-policy learning (decorrelates samples)
      - Target networks with soft updates for stability
      - Gaussian exploration noise that decays over training
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
                 noise_init_ratio=0.5,
                 noise_min_ratio=0.1,
                 noise_decay_steps=100_000):
        # (2) Store hyperparameters.
        self.gamma              = gamma
        self.tau                = tau
        self.batch_size         = batch_size
        self.warmup_steps       = warmup_steps
        self.policy_max_grad_norm = policy_max_grad_norm
        self.value_max_grad_norm  = value_max_grad_norm

        # (3) Build the online policy model mu(s; phi) — the actor network.
        self.online_policy_model = FCDP(obs_dim, action_bounds, hidden_dims)
        # (4) Build the target policy model mu(s; phi^-) — a copy of the actor used for stable targets.
        self.target_policy_model = copy.deepcopy(self.online_policy_model)

        # (5) Build the online Q-value model Q(s, a; theta) — the critic network.
        self.online_value_model = FCQV(obs_dim, act_dim, hidden_dims)
        # (6) Build the target Q-value model Q(s, a; theta^-) — a copy used for stable targets.
        self.target_value_model = copy.deepcopy(self.online_value_model)

        # (7) Target networks are never directly optimized — they are only updated via soft copy.
        for p in self.target_policy_model.parameters():
            p.requires_grad = False
        for p in self.target_value_model.parameters():
            p.requires_grad = False

        # (8) Build optimizers for the online policy and value networks.
        self.policy_optimizer = optim.Adam(self.online_policy_model.parameters(), lr=actor_lr)
        self.value_optimizer  = optim.Adam(self.online_value_model.parameters(), lr=critic_lr)

        # (9) Build the replay buffer and the noise strategy for exploration.
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)
        self.noise_strategy = NormalNoiseDecayStrategy(
            action_bounds, noise_init_ratio, noise_min_ratio, noise_decay_steps)

        self.total_steps = 0

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, state, training=True):
        """
        During training, add exploration noise to the deterministic action.
        During evaluation, use the clean deterministic policy output.
        """
        if training:
            max_exploration = self.total_steps < self.warmup_steps
            return self.noise_strategy.select_action(
                self.online_policy_model, state, max_exploration)
        else:
            return self.online_policy_model.select_action(state)

    def store(self, state, action, reward, next_state, done):
        """Store the transition in the replay buffer and advance the noise decay schedule."""
        self.replay_buffer.store(state, action, reward, next_state, done)
        self.noise_strategy.update()
        self.total_steps += 1

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self):
        # (12) Only start optimizing once the buffer has enough samples.
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # (13) Pull out a mini-batch of transitions from the replay buffer.
        states, actions, rewards, next_states, is_terminals = \
            self.replay_buffer.sample(self.batch_size)

        # --- Value network update ---

        # (14) Calculate targets using the predicted max value of the next state.
        #      DDPG target: y = r + gamma * Q_target(s', mu_target(s'))
        #      Using target networks (theta^-, phi^-) for stable TD targets.
        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp    = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa   = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)

        # (15) Get the current Q-value predictions and calculate the TD error.
        q_sa     = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa.detach()
        # (16) Value loss: 0.5 * MSE = 0.5 * E[(Q(s,a) - y)^2].
        value_loss = td_error.pow(2).mul(0.5).mean()

        # (17) Optimize the value network: zero gradients, backprop, clip, step.
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_value_model.parameters(), self.value_max_grad_norm)
        self.value_optimizer.step()

        # --- Policy network update ---

        # (18) The policy loss is the negative expected Q-value under the current policy:
        #      L(phi) = -E_s[Q(s, mu(s; phi); theta)]
        #      We maximize Q by minimizing its negative.
        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s    = self.online_value_model(states, argmax_a_q_s)
        policy_loss  = -max_a_q_s.mean()

        # (19) Optimize the policy network: zero gradients, backprop, clip, step.
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_policy_model.parameters(), self.policy_max_grad_norm)
        self.policy_optimizer.step()

        # (20) Soft-update both target networks using Polyak averaging:
        #      theta^- <- tau * theta + (1 - tau) * theta^-
        #      A small tau (e.g. 0.005) makes the target networks move slowly, improving stability.
        self._soft_update(self.online_value_model,  self.target_value_model)
        self._soft_update(self.online_policy_model, self.target_policy_model)

        return value_loss.item(), policy_loss.item()

    def _soft_update(self, online, target):
        """Polyak averaging: blend online network weights into the target network."""
        for online_p, target_p in zip(online.parameters(), target.parameters()):
            target_p.data.copy_(
                self.tau * online_p.data + (1.0 - self.tau) * target_p.data)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path):
        """Save all network and optimizer states to disk for later resuming."""
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
        """Restore all network and optimizer states from a checkpoint file."""
        ckpt = torch.load(path, map_location='cpu')
        self.online_policy_model.load_state_dict(ckpt['online_policy'])
        self.target_policy_model.load_state_dict(ckpt['target_policy'])
        self.online_value_model.load_state_dict(ckpt['online_value'])
        self.target_value_model.load_state_dict(ckpt['target_value'])
        self.policy_optimizer.load_state_dict(ckpt['policy_opt'])
        self.value_optimizer.load_state_dict(ckpt['value_opt'])
        self.total_steps = ckpt['total_steps']
