import numpy as np
import torch
import torch.optim as optim

from rl_continuous.networks.policy_nets import FCGP
from rl_continuous.networks.value_nets import FCV
from rl_continuous.utils.episode_buffer import EpisodeBuffer


class PPO:
    """
    PPO stands for Proximal Policy Optimization (Schulman et al., 2017).
    PPO is an ON-POLICY actor-critic algorithm (unlike DDPG/TD3/SAC which are off-policy).
    It collects a batch of on-policy experience, then optimizes for multiple epochs
    on mini-batches from that batch — reusing data without large policy degradation.

    The key idea (Morales, Ch.12; Landers, ppo.html):
      Instead of constraining the KL divergence (like TRPO), PPO clips the probability ratio:
        r_t(theta) = pi(a|s; theta) / pi(a|s; phi^-)    (phi^- = old policy parameters)
      PPO clipped objective: L^CLIP(theta) = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
      This prevents excessively large policy updates while allowing multiple gradient steps.

    PPO also clips the value function loss and uses GAE for advantage estimation.
    Early stopping based on KL divergence (policy) and MSE (value) avoids over-optimization.
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 action_bounds,
                 hidden_dims=(256, 256),
                 actor_lr=3e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=0.95,
                 policy_clip_range=0.2,
                 value_clip_range=0.2,
                 policy_optimization_epochs=10,
                 value_optimization_epochs=10,
                 policy_sample_ratio=0.8,
                 value_sample_ratio=0.8,
                 entropy_loss_weight=0.01,
                 policy_max_grad_norm=0.5,
                 value_max_grad_norm=0.5,
                 policy_stopping_kl=0.02,
                 value_stopping_mse=25.0):
        # (2) Store hyperparameters.
        self.gamma                      = gamma
        self.policy_clip_range          = policy_clip_range
        self.value_clip_range           = value_clip_range
        self.policy_optimization_epochs = policy_optimization_epochs
        self.value_optimization_epochs  = value_optimization_epochs
        self.policy_sample_ratio        = policy_sample_ratio
        self.value_sample_ratio         = value_sample_ratio
        self.entropy_loss_weight        = entropy_loss_weight
        self.policy_max_grad_norm       = policy_max_grad_norm
        self.value_max_grad_norm        = value_max_grad_norm
        self.policy_stopping_kl         = policy_stopping_kl
        self.value_stopping_mse         = value_stopping_mse

        # (3) Build the stochastic Gaussian policy pi(a|s; phi) — the actor.
        self.policy_model = FCGP(obs_dim, action_bounds, hidden_dims)
        # (4) Build the state-value network V(s; psi) — the critic.
        self.value_model  = FCV(obs_dim, hidden_dims)

        # (5) Build separate optimizers for actor and critic.
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=actor_lr)
        self.value_optimizer  = optim.Adam(self.value_model.parameters(),  lr=critic_lr)

        # (6) EpisodeBuffer stores on-policy trajectories and computes GAE advantages.
        self.episode_buffer = EpisodeBuffer(gamma=gamma, tau=tau)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, state, training=True):
        """
        During training, sample a stochastic action and return also its log-probability
        and the critic's value estimate — needed to fill the episode buffer.
        During evaluation, use the deterministic greedy action.
        """
        if training:
            action, logpa, _, _, _ = self.policy_model.full_pass(state)
            value = self.value_model(state)
            return (action.detach().cpu().numpy().squeeze(0),
                    logpa.detach().cpu().numpy().squeeze(),
                    value.detach().cpu().numpy().squeeze())
        else:
            return self.policy_model.select_greedy_action(state)

    def store(self, state, action, reward, value, logpa, done):
        """Store the transition in the episode buffer."""
        self.episode_buffer.store(state, action, reward, value, logpa, done)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self, last_value=0.0):
        """
        Retrieve all stored experience from the episode buffer.
        GAE advantages and discounted returns are computed inside EpisodeBuffer.get().
        """
        states, actions, returns, gaes, logpas = self.episode_buffer.get(last_value)
        n_samples = len(actions)

        # (10) Get the current value predictions for the full batch (used for clipped value loss).
        values = self.value_model(states).detach()

        # (11) Normalize advantages to mean=0, std=1.
        #      This stabilizes learning by preventing very large or small policy gradients.
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        # --- Policy optimization (multiple epochs on mini-batches) ---

        policy_loss_epoch = 0.0
        for _ in range(self.policy_optimization_epochs):
            # (12) Sub-sample a random mini-batch from the collected experience.
            batch_size = max(1, int(self.policy_sample_ratio * n_samples))
            batch_idxs = np.random.choice(n_samples, batch_size, replace=False)

            states_b  = states[batch_idxs]
            actions_b = actions[batch_idxs]
            gaes_b    = gaes[batch_idxs]
            logpas_b  = logpas[batch_idxs]

            # (13) Get log-probabilities and entropies under the NEW (current) policy.
            new_actions, new_logpas, _, _, _ = self.policy_model.full_pass(states_b)
            # (14) Recompute log-probabilities for the OLD actions stored in the buffer.
            _, logpas_pred = self._eval_logpa(states_b, actions_b)
            entropies = -new_logpas

            # (15) Compute the probability ratio r_t(theta) = pi_new(a|s) / pi_old(a|s).
            #      In log space: r_t = exp(log_pi_new - log_pi_old).
            ratios  = (logpas_pred - logpas_b).exp()
            pi_obj  = gaes_b * ratios
            # (16) Clip the ratio to [1-eps, 1+eps] to limit the policy update size.
            #      PPO objective: L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
            pi_obj_clipped = gaes_b * ratios.clamp(
                1.0 - self.policy_clip_range, 1.0 + self.policy_clip_range)

            # (17) Take the minimum between unclipped and clipped objectives (pessimistic bound).
            policy_loss  = -torch.min(pi_obj, pi_obj_clipped).mean()
            # (18) Add the entropy bonus to encourage exploration (weighted by entropy_loss_weight).
            entropy_loss = entropies.mean() * self.entropy_loss_weight
            total_policy_loss = policy_loss + entropy_loss

            # (19) Optimize the policy network.
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.policy_max_grad_norm)
            self.policy_optimizer.step()
            policy_loss_epoch += total_policy_loss.item()

            # (20) Early stopping: if the KL divergence between old and new policy exceeds
            #      the threshold, stop optimizing to prevent the policy from changing too much.
            with torch.no_grad():
                _, all_logpas_pred = self._eval_logpa(states, actions)
                kl = (logpas - all_logpas_pred).mean()
            if kl.item() > self.policy_stopping_kl:
                break

        # --- Value optimization (multiple epochs on mini-batches) ---

        value_loss_epoch = 0.0
        for _ in range(self.value_optimization_epochs):
            # (21) Sub-sample a random mini-batch for the value function update.
            batch_size = max(1, int(self.value_sample_ratio * n_samples))
            batch_idxs = np.random.choice(n_samples, batch_size, replace=False)

            states_b  = states[batch_idxs]
            returns_b = returns[batch_idxs]
            values_b  = values[batch_idxs]

            # (22) Compute value predictions and standard MSE loss.
            values_pred = self.value_model(states_b)
            v_loss = (values_pred - returns_b).pow(2)

            # (23) Clip the value prediction change relative to the old estimate.
            #      This prevents the value function from changing too drastically in one step.
            values_pred_clipped = values_b + (values_pred - values_b).clamp(
                -self.value_clip_range, self.value_clip_range)
            v_loss_clipped = (values_pred_clipped - returns_b).pow(2)

            # (24) Use the maximum of standard and clipped losses (conservative estimate).
            value_loss = torch.max(v_loss, v_loss_clipped).mul(0.5).mean()

            # (25) Optimize the value network.
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_model.parameters(), self.value_max_grad_norm)
            self.value_optimizer.step()
            value_loss_epoch += value_loss.item()

            # (26) Early stopping: if the value function MSE worsens beyond a threshold,
            #      stop to prevent the critic from diverging.
            with torch.no_grad():
                all_values_pred = self.value_model(states)
            mse = (values - all_values_pred).pow(2).mul(0.5).mean()
            if mse.item() > self.value_stopping_mse:
                break

        return value_loss_epoch, policy_loss_epoch

    def _eval_logpa(self, states, actions):
        """
        Evaluate the log-probability of given (state, action) pairs under the current policy.
        This is needed to compute the probability ratio r_t = pi_new / pi_old.
        """
        _, mean, log_std = self._get_dist_params(states)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        # (28) Inverse-tanh to recover the pre-squashing action from the stored action.
        #      Stored actions are already rescaled; we un-rescale then un-tanh them.
        env_min = self.policy_model.env_min
        env_max = self.policy_model.env_max
        normalized = (actions - env_min) / (env_max - env_min) * 2 - 1
        normalized = normalized.clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh = torch.atanh(normalized)
        # (29) Compute log-probability with the tanh change-of-variables correction.
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log((1 - normalized.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return normalized, log_prob

    def _get_dist_params(self, states):
        """Helper: get distribution parameters (mean, log_std) from the policy network."""
        mean, log_std = self.policy_model.forward(states)
        return None, mean, log_std

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            'policy':     self.policy_model.state_dict(),
            'value':      self.value_model.state_dict(),
            'policy_opt': self.policy_optimizer.state_dict(),
            'value_opt':  self.value_optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.policy_model.load_state_dict(ckpt['policy'])
        self.value_model.load_state_dict(ckpt['value'])
        self.policy_optimizer.load_state_dict(ckpt['policy_opt'])
        self.value_optimizer.load_state_dict(ckpt['value_opt'])
