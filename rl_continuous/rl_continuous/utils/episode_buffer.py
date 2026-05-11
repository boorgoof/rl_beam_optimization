import numpy as np
import torch


class EpisodeBuffer:
    """
    EpisodeBuffer stores on-policy trajectories collected by PPO workers.
    Unlike ReplayBuffer, data here is NOT reused across updates — it is discarded
    after each policy optimization. The buffer also computes:
      - discounted returns G_t
      - GAE advantages A^GAE(gamma, lambda) (Schulman et al., 2015)
    GAE formula: A^GAE_t = sum_{l=0}^inf (gamma*lambda)^l * delta_{t+l}
    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  (TD residual)
    """
    def __init__(self, gamma=0.99, tau=0.95):
        # (2) gamma is the discount factor for returns; tau (lambda in the GAE paper) is
        #     the GAE smoothing parameter that trades off bias vs. variance:
        #     tau=0 -> 1-step TD (low variance, high bias)
        #     tau=1 -> full Monte Carlo return (high variance, low bias)
        self.gamma = gamma
        self.tau   = tau
        self._reset()

    def _reset(self):
        # (3) Internal storage lists — cleared after each policy update.
        self._states    = []
        self._actions   = []
        self._rewards   = []
        self._values    = []
        self._logpas    = []
        self._terminals = []

    def store(self, state, action, reward, value, logpa, done):
        """
        Store a single time-step transition along with the critic value estimate
        and the log-probability of the action (needed for the PPO ratio r_t(theta)).
        """
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._logpas.append(logpa)
        self._terminals.append(done)

    def get(self, last_value=0.0):
        """
        Retrieve all stored data, compute returns and GAE advantages, then reset.
        last_value is the bootstrap value V(s_T) at the end of the trajectory
        (0 if the episode terminated, V(s_T) if truncated).
        """
        states   = np.array(self._states,  dtype=np.float32)
        actions  = np.array(self._actions, dtype=np.float32)
        rewards  = np.array(self._rewards, dtype=np.float32)
        values   = np.array(self._values,  dtype=np.float32)
        logpas   = np.array(self._logpas,  dtype=np.float32)

        T = len(rewards)

        # (6) Compute discounted returns G_t = r_t + gamma*r_{t+1} + ... + gamma^{T-t-1}*r_{T-1}
        #     + gamma^{T-t} * last_value, used as targets for the value network.
        returns = np.zeros(T, dtype=np.float32)
        G = last_value
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G * (1.0 - self._terminals[t])
            returns[t] = G

        # (7) Compute GAE advantages using the TD residuals:
        #     delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        #     A^GAE_t = delta_t + (gamma*tau)*delta_{t+1} + (gamma*tau)^2*delta_{t+2} + ...
        values_extended = np.append(values, last_value)
        gaes = np.zeros(T, dtype=np.float32)
        gae  = 0.0
        for t in reversed(range(T)):
            not_terminal = 1.0 - self._terminals[t]
            delta = rewards[t] + self.gamma * values_extended[t + 1] * not_terminal - values[t]
            # (8) Accumulate the discounted GAE sum backwards.
            gae   = delta + self.gamma * self.tau * not_terminal * gae
            gaes[t] = gae

        # (9) Convert everything to tensors for use in the optimizer.
        states_t  = torch.tensor(states)
        actions_t = torch.tensor(actions)
        returns_t = torch.tensor(returns)
        gaes_t    = torch.tensor(gaes)
        logpas_t  = torch.tensor(logpas)

        self._reset()
        return states_t, actions_t, returns_t, gaes_t, logpas_t
