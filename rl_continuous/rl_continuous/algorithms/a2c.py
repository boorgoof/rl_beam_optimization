import torch
import torch.optim as optim

from rl_continuous.networks.policy_nets import FCAC
from rl_continuous.utils.episode_buffer import EpisodeBuffer


class A2C:
    """
    A2C — Advantage Actor-Critic, synchronous version (Morales, Ch.11).
    Adapted here for continuous action spaces using a Gaussian policy.

    Key design choices (Morales, Ch.11):
      - Shared backbone network (FCAC) for both policy and value heads.
        A single forward pass produces the action distribution AND V(s).
      - ONE gradient step per episode rollout (no multiple epochs like PPO).
      - Combined loss: policy_weight * L_pi + value_weight * L_v + entropy_weight * L_H
          L_pi = -E[A(s,a) * log_pi(a|s)]       (advantage-weighted log-prob)
          L_v  = 0.5 * E[(V(s) - G_t)^2]        (MSE on discounted returns)
          L_H  = -E[H(pi(.|s))]                  (maximize entropy for exploration)
      - GAE (Generalized Advantage Estimation) for advantage computation.
      - Advantage normalization for gradient stability.

    Default hyperparameters follow Morales Ch.11:
      lr=0.002, value_loss_weight=0.6, entropy_loss_weight=0.001, max_grad_norm=1.0
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 action_bounds,
                 hidden_dims=(256, 256),
                 lr=0.002,
                 gamma=0.99,
                 tau=0.95,
                 policy_loss_weight=1.0,
                 value_loss_weight=0.6,
                 entropy_loss_weight=0.001,
                 max_grad_norm=1.0):
        # (1) Store hyperparameters.
        self.gamma               = gamma
        self.policy_loss_weight  = policy_loss_weight
        self.value_loss_weight   = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.max_grad_norm       = max_grad_norm

        # (2) Build the shared actor-critic network (Morales FCAC architecture).
        self.ac_model = FCAC(obs_dim, action_bounds, hidden_dims)

        # (3) Single optimizer for the shared network (Morales Ch.11: lr=0.002).
        self.optimizer = optim.Adam(self.ac_model.parameters(), lr=lr)

        # (4) EpisodeBuffer stores on-policy trajectories and computes GAE advantages.
        self.episode_buffer = EpisodeBuffer(gamma=gamma, tau=tau)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, state, training=True):
        """
        During training, sample a stochastic action and return (action, logpa, value)
        — all three are needed to fill the EpisodeBuffer.
        During evaluation, use the deterministic greedy action.
        """
        if training:
            return self.ac_model.select_action(state)
        else:
            return self.ac_model.select_greedy_action(state)

    def store(self, state, action, reward, value, logpa, done):
        """Store the transition in the episode buffer."""
        self.episode_buffer.store(state, action, reward, value, logpa, done)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self, last_value=0.0):
        """
        ONE gradient step on the full episode rollout (Morales Ch.11).
        Combined loss: policy_weight * L_pi + value_weight * L_v + entropy_weight * L_H
        """
        states, actions, returns, gaes, logpas = self.episode_buffer.get(last_value)

        # (5) Normalize advantages (mean=0, std=1) for gradient stability.
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        # (6) Forward pass through the shared network to get current distribution params and values.
        mean, log_std, values_pred = self.ac_model.forward(states)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        # (7) Recover pre-tanh actions from stored (rescaled) actions to recompute log_prob.
        env_min    = self.ac_model.env_min
        env_max    = self.ac_model.env_max
        normalized = (actions - env_min) / (env_max - env_min) * 2 - 1
        normalized = normalized.clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh   = torch.atanh(normalized)

        # (8) Log-prob with tanh change-of-variables correction.
        log_prob  = dist.log_prob(pre_tanh)
        log_prob -= torch.log((1 - normalized.pow(2)) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)

        # (9) Entropy: H(pi) approximated as -log_pi (Monte Carlo estimate).
        entropies = -log_prob

        # --- Policy loss: L_pi = -E[A(s,a) * log_pi(a|s)] ---
        policy_loss = -(gaes * log_prob).mean()

        # --- Value loss: L_v = 0.5 * E[(V(s) - G_t)^2] ---
        value_loss = (values_pred - returns).pow(2).mul(0.5).mean()

        # --- Entropy loss: L_H = -E[H(pi)] (minimizing this maximizes entropy) ---
        entropy_loss = -entropies.mean()

        # (10) Combined loss (Morales Ch.11: weights 1.0, 0.6, 0.001).
        total_loss = (self.policy_loss_weight  * policy_loss
                    + self.value_loss_weight   * value_loss
                    + self.entropy_loss_weight * entropy_loss)

        # (11) Single backward pass and gradient step on the shared network.
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return value_loss.item(), policy_loss.item()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path):
        """Save the shared network and optimizer state to disk."""
        torch.save({
            'ac_model':  self.ac_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Restore the shared network and optimizer state from a checkpoint."""
        ckpt = torch.load(path, map_location='cpu')
        self.ac_model.load_state_dict(ckpt['ac_model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
