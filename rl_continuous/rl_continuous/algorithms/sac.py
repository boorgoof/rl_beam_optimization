import copy
import numpy as np
import torch
import torch.optim as optim

from rl_continuous.networks.policy_nets import FCGP
from rl_continuous.networks.value_nets import FCQV
from rl_continuous.utils.replay_buffer import ReplayBuffer


class SAC:
    """
    SAC stands for Soft Actor-Critic (Haarnoja et al., 2018).
    SAC is an off-policy algorithm that trains a STOCHASTIC policy (unlike DDPG/TD3)
    and maximizes a entropy-augmented objective (Morales, Ch.12; Landers, sac.html):
      J(pi) = E[sum_t gamma^t (r_t + alpha * H(pi(.|s_t)))]
    This encourages the policy to be as random as possible while still maximizing reward,
    which improves exploration and robustness.

    SAC uses three key components:
      - Twin Q-networks (from TD3): min(Q_a, Q_b) to reduce overestimation
      - Stochastic Gaussian policy with reparameterization trick (FCGP)
      - Automatic entropy tuning: alpha is optimized to meet a target entropy H0
        The entropy constraint: E[-log pi(a|s)] >= H0, solved via Lagrangian duality.
        H0 = -dim(action_space)  (heuristic from Haarnoja et al.)
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 action_bounds,
                 hidden_dims=(256, 256),
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 buffer_size=int(1e6),
                 batch_size=256,
                 warmup_steps=1000,
                 policy_max_grad_norm=float('inf'),
                 value_max_grad_norm=float('inf')):
        # (2) Store hyperparameters.
        self.gamma                = gamma
        self.tau                  = tau
        self.batch_size           = batch_size
        self.warmup_steps         = warmup_steps
        self.policy_max_grad_norm = policy_max_grad_norm
        self.value_max_grad_norm  = value_max_grad_norm

        # (3) Build the stochastic Gaussian policy pi(a|s; phi).
        #     FCGP also holds logalpha (the learnable entropy coefficient) and target_entropy.
        self.policy_model = FCGP(obs_dim, action_bounds, hidden_dims)

        # (4) Build two independent online Q-networks Q_a and Q_b (s, a; theta).
        #     SAC uses two separate FCQV instances (not FCTQV), so each has its own optimizer.
        self.online_value_model_a = FCQV(obs_dim, act_dim, hidden_dims)
        self.online_value_model_b = FCQV(obs_dim, act_dim, hidden_dims)
        # (5) Build two corresponding target Q-networks for stable TD targets.
        self.target_value_model_a = copy.deepcopy(self.online_value_model_a)
        self.target_value_model_b = copy.deepcopy(self.online_value_model_b)

        # (6) Target networks are never directly optimized.
        for p in self.target_value_model_a.parameters():
            p.requires_grad = False
        for p in self.target_value_model_b.parameters():
            p.requires_grad = False

        # (7) Build separate optimizers for policy, each critic, and alpha.
        self.policy_optimizer   = optim.Adam(self.policy_model.parameters(), lr=actor_lr)
        self.value_optimizer_a  = optim.Adam(self.online_value_model_a.parameters(), lr=critic_lr)
        self.value_optimizer_b  = optim.Adam(self.online_value_model_b.parameters(), lr=critic_lr)
        # (8) Alpha optimizer: optimizes logalpha to satisfy the entropy constraint.
        #     We optimize log(alpha) instead of alpha directly to keep alpha > 0.
        self.alpha_optimizer    = optim.Adam([self.policy_model.logalpha], lr=alpha_lr)

        # (9) Build the replay buffer.
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)
        self.total_steps   = 0

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, state, training=True):
        """
        During training, sample a stochastic action for exploration.
        During evaluation, use the greedy (deterministic mean) action.
        During warm-up, sample uniformly within action bounds for initial exploration.
        """
        if training and self.total_steps >= self.warmup_steps:
            return self.policy_model.select_action(state)
        elif training:
            # (11) Warm-up: sample uniformly within action bounds for initial exploration.
            env_min = self.policy_model.env_min.numpy()
            env_max = self.policy_model.env_max.numpy()
            return np.random.uniform(env_min, env_max)
        else:
            return self.policy_model.select_greedy_action(state)

    def store(self, state, action, reward, next_state, done):
        """Store the transition in the replay buffer."""
        self.replay_buffer.store(state, action, reward, next_state, done)
        self.total_steps += 1

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self):
        # (13) Only start optimizing once the buffer has enough samples.
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None

        # (14) Pull out a mini-batch of transitions from the replay buffer.
        states, actions, rewards, next_states, is_terminals = \
            self.replay_buffer.sample(self.batch_size)

        # (15) Get the current alpha (entropy coefficient) value from logalpha.
        alpha = self.policy_model.logalpha.exp().detach()

        # --- Critic update ---

        # (16) Sample next actions and their log-probabilities from the current policy.
        with torch.no_grad():
            next_actions, logpi_sp, _, _, _ = self.policy_model.full_pass(next_states)
            # (17) Get Q-value estimates from both target networks for the next state.
            q_sp_a = self.target_value_model_a(next_states, next_actions)
            q_sp_b = self.target_value_model_b(next_states, next_actions)
            # (18) SAC target: y = r + gamma * (min(Q_a, Q_b)(s', a') - alpha * log_pi(a'|s'))
            #      The entropy term - alpha * log_pi acts as a soft value bonus that
            #      encourages visiting states with high-entropy actions.
            min_q_sp  = torch.min(q_sp_a, q_sp_b)
            target_q_sa = rewards + self.gamma * (min_q_sp - alpha * logpi_sp) * (1 - is_terminals)

        # (19) Compute TD errors for both online critics and optimize them independently.
        q_sa_a     = self.online_value_model_a(states, actions)
        q_sa_b     = self.online_value_model_b(states, actions)
        value_loss_a = (q_sa_a - target_q_sa).pow(2).mul(0.5).mean()
        value_loss_b = (q_sa_b - target_q_sa).pow(2).mul(0.5).mean()

        # (20) Optimize critic A.
        self.value_optimizer_a.zero_grad()
        value_loss_a.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_value_model_a.parameters(), self.value_max_grad_norm)
        self.value_optimizer_a.step()

        # (21) Optimize critic B.
        self.value_optimizer_b.zero_grad()
        value_loss_b.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_value_model_b.parameters(), self.value_max_grad_norm)
        self.value_optimizer_b.step()

        # --- Policy update ---

        # (22) Sample actions and log-probs from the current policy for the policy gradient.
        current_actions, logpi_s, _, _, _ = self.policy_model.full_pass(states)
        # (23) Get Q-value estimates for the current actions from both online critics.
        q_s_a = self.online_value_model_a(states, current_actions)
        q_s_b = self.online_value_model_b(states, current_actions)
        min_q_s = torch.min(q_s_a, q_s_b)
        # (24) SAC policy loss: L(phi) = E[alpha * log_pi(a|s) - min(Q_a, Q_b)(s, a)]
        #      We maximize the Q-value while penalizing low-entropy policies.
        #      Equivalently, we minimize the KL divergence between pi and exp(Q/alpha).
        policy_loss = (alpha * logpi_s - min_q_s).mean()

        # (25) Optimize the policy network.
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.policy_max_grad_norm)
        self.policy_optimizer.step()

        # --- Alpha (entropy coefficient) update ---

        # (26) Alpha loss: J(alpha) = E[-alpha * log_pi(a|s) - alpha * H0]
        #      We optimize alpha so that the policy entropy stays close to the target entropy H0.
        #      H0 = -dim(action_space) is a heuristic (Haarnoja et al., 2018; Landers, sac.html).
        alpha_loss = -(self.policy_model.logalpha *
                       (logpi_s + self.policy_model.target_entropy).detach()).mean()

        # (27) Optimize logalpha.
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # (28) Soft-update both target Q-networks using Polyak averaging.
        self._soft_update(self.online_value_model_a, self.target_value_model_a)
        self._soft_update(self.online_value_model_b, self.target_value_model_b)

        value_loss = (value_loss_a + value_loss_b).item() / 2
        return value_loss, policy_loss.item(), alpha_loss.item()

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
            'policy':          self.policy_model.state_dict(),
            'online_value_a':  self.online_value_model_a.state_dict(),
            'online_value_b':  self.online_value_model_b.state_dict(),
            'target_value_a':  self.target_value_model_a.state_dict(),
            'target_value_b':  self.target_value_model_b.state_dict(),
            'policy_opt':      self.policy_optimizer.state_dict(),
            'value_opt_a':     self.value_optimizer_a.state_dict(),
            'value_opt_b':     self.value_optimizer_b.state_dict(),
            'alpha_opt':       self.alpha_optimizer.state_dict(),
            'total_steps':     self.total_steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.policy_model.load_state_dict(ckpt['policy'])
        self.online_value_model_a.load_state_dict(ckpt['online_value_a'])
        self.online_value_model_b.load_state_dict(ckpt['online_value_b'])
        self.target_value_model_a.load_state_dict(ckpt['target_value_a'])
        self.target_value_model_b.load_state_dict(ckpt['target_value_b'])
        self.policy_optimizer.load_state_dict(ckpt['policy_opt'])
        self.value_optimizer_a.load_state_dict(ckpt['value_opt_a'])
        self.value_optimizer_b.load_state_dict(ckpt['value_opt_b'])
        self.alpha_optimizer.load_state_dict(ckpt['alpha_opt'])
        self.total_steps = ckpt['total_steps']
