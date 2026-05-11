import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FCDP(nn.Module):
    """
    FCDP stands for Fully Connected Deterministic Policy. It outputs a single deterministic action.

    @param input_dim: dimension of the state space (obs_dim)
    @param action_bounds: tuple of (min_action, max_action) for rescaling the tanh output to the environment's action range.
    @param hidden_dims: tuple specifying the number of neurons in each hidden layer (default: (256, 256), that is two hidden layers with 256 neurons each)
    @param activation_fc: activation function used by the network (default: F.relu)
    @param out_activation_fc: activation function for the output layer (default: F.tanh, to bound actions in (-1,1), then rescaled to env action range using action_bounds parameter)

    """
    def __init__(self,
                input_dim,
                action_bounds,
                hidden_dims=(256, 256),
                activation_fc=F.relu,
                out_activation_fc=F.tanh):
        
        super(FCDP, self).__init__()

    
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc

        # Convert action bounds to tensors so that we can rescale the tanh output.
        self.env_min = torch.tensor(action_bounds[0], dtype=torch.float32)
        self.env_max = torch.tensor(action_bounds[1], dtype=torch.float32)

        # layer definitions
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], len(self.env_max))

        # rescaling constants: tanh outputs in (-1,1), then we want to map to (env_min, env_max).
        self.rescale_fn = lambda x: (((x + 1) / 2) * (self.env_max - self.env_min) + self.env_min)

    def _format(self, state):
        """Function to be sure that state is a tensor of the expected shape i.e. (batch_size, feature_obs_dim)."""

        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        """Pass the state through the network and return the action.

        @param state: the state input of shape (batch_size, obs_dim) after formatting
        @return: the action output of shape (batch_size, act_dim) after rescaling to env action range
        """

        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.out_activation_fc(self.output_layer(x))

        return self.rescale_fn(x)

    def select_action(self, state):
        """
        Compute the deterministic action for the given state and return it
        as a NumPy array suitable for environment interaction, i.e convert from (batch_size, act_dim), to (act_dim,)
        """
        return self.forward(state).detach().cpu().numpy().squeeze(0) # squeeze remove only the batch dimension: (1, act_dim) -> (act_dim,)



class FCGP(nn.Module):
    """
    FCGP stands for Fully Connected Gaussian Policy. It is used by SAC and PPO. It outputs a stochastic action by sampling from N(mu, sigma^2).
    """

    def __init__(self,
                input_dim,
                action_bounds,
                hidden_dims=(256, 256),
                activation_fc=F.relu,
                log_std_min=-20,
                log_std_max=0):
                 
        super(FCGP, self).__init__()

        # (15) Store bounds, activation, and log_std clipping range.
        self.activation_fc = activation_fc
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # (16) Convert action bounds to tensors for rescaling and entropy correction.
        self.env_min = torch.tensor(action_bounds[0], dtype=torch.float32)
        self.env_max = torch.tensor(action_bounds[1], dtype=torch.float32)

        # (17) The __init__ function creates a linear connection between input and first hidden layer.
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        # (18) Then, it creates connections across all hidden layers.
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        # (19) The policy outputs two separate heads: mean and log standard deviation.
        self.output_layer_mean = nn.Linear(hidden_dims[-1], len(self.env_max))
        self.output_layer_log_std = nn.Linear(hidden_dims[-1], len(self.env_max))

        # (20) Target entropy: heuristic = -dim(action_space), as suggested by Landers/SAC paper.
        #      J(alpha) = E[-alpha * log_pi - alpha * H0], H0 = target_entropy.
        self.target_entropy = -np.prod(self.env_max.shape)
        # (21) Learnable log-alpha for automatic entropy tuning (SAC).
        #      We optimize alpha to satisfy the entropy constraint: E[-log_pi] >= H0.
        self.logalpha = nn.Parameter(torch.zeros(1))

        # (22) Precompute rescaling constants to map tanh output from (-1,1) to (env_min, env_max).
        self.rescale_fn = lambda x: (((x + 1) / 2) * (self.env_max - self.env_min) + self.env_min)

    def _format(self, state):
        """Make sure the state is of the type of variable and shape we expect."""
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        """Pass the state through the shared body (input + hidden layers) and return mean and log_std."""
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        # (25) Compute mean and log_std from the two separate output heads.
        mean = self.output_layer_mean(x)
        log_std = self.output_layer_log_std(x)
        # (26) Clamp log_std to [-20, 0] to prevent numerical instabilities.
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def full_pass(self, state):
        """
        full_pass is a handy function to obtain log-probabilities, actions,
        and entropy — everything needed for training.
        Returns (action, log_prob, tanh_action, mean, log_std).
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # (28) Sample pre-squashing action using the reparameterization trick: a = mu + eps * sigma.
        dist = torch.distributions.Normal(mean, std)
        pre_tanh_action = dist.rsample()
        # (29) Apply tanh squashing to bound the action, then rescale to env action range.
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.rescale_fn(tanh_action)
        # (30) Compute log-probability with the change-of-variables correction for tanh squashing:
        #      log_pi(a|s) = log_pi(u|s) - sum(log(1 - tanh(u)^2))
        #      This is the correction term from the SAC paper (Haarnoja et al., 2018).
        log_prob = dist.log_prob(pre_tanh_action)
        log_prob -= torch.log((1 - tanh_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, tanh_action, mean, log_std

    def select_action(self, state):
        """Helper: sample a stochastic action of shape (act_dim,) for environment interaction."""
        action, _, _, _, _ = self.full_pass(state)
        return action.detach().cpu().numpy().squeeze(0)

    def select_greedy_action(self, state):
        """Helper: select the deterministic (greedy) action by using the mean directly."""
        mean, _ = self.forward(state)
        action = self.rescale_fn(torch.tanh(mean))
        return action.detach().cpu().numpy().squeeze(0)


class FCAC(nn.Module):
    """
    FCAC stands for Fully Connected Actor-Critic (shared network).
    Used by A2C (Morales, Ch.11), adapted here for continuous action spaces.

    Shares a hidden-layer backbone between the policy and value heads, so a
    single forward pass produces both the action distribution and V(s).
    This can be more compute-efficient than keeping two separate networks.

    Policy head : two linear heads outputting mean and log_std of a Gaussian.
    Value head  : one linear head outputting the scalar V(s).
    """
    def __init__(self,
                 input_dim,
                 action_bounds,
                 hidden_dims=(256, 256),
                 activation_fc=F.relu,
                 log_std_min=-20,
                 log_std_max=0):
        super(FCAC, self).__init__()
        self.activation_fc = activation_fc
        self.log_std_min    = log_std_min
        self.log_std_max    = log_std_max
        self.env_min = torch.tensor(action_bounds[0], dtype=torch.float32)
        self.env_max = torch.tensor(action_bounds[1], dtype=torch.float32)

        # Shared backbone.
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Policy head: Gaussian mean and log_std.
        self.policy_output_layer_mean    = nn.Linear(hidden_dims[-1], len(self.env_max))
        self.policy_output_layer_log_std = nn.Linear(hidden_dims[-1], len(self.env_max))

        # Value head: scalar V(s).
        self.value_output_layer = nn.Linear(hidden_dims[-1], 1)

        self.rescale_fn = lambda x: (((x + 1) / 2) * (self.env_max - self.env_min) + self.env_min)

    def _format(self, state):
        """Make sure the state is of the type of variable and shape we expect."""
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def _shared_forward(self, state):
        """Pass the state through the shared backbone and return the last hidden representation."""
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return x

    def forward(self, state):
        """Return (mean, log_std, value) from a single shared forward pass."""
        x = self._shared_forward(state)
        mean    = self.policy_output_layer_mean(x)
        log_std = self.policy_output_layer_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        value   = self.value_output_layer(x)
        return mean, log_std, value

    def full_pass(self, state):
        """
        Single forward pass returning everything needed for A2C training:
        (action, log_prob, value, tanh_action, mean, log_std).
        """
        mean, log_std, value = self.forward(state)
        std = log_std.exp()
        dist            = torch.distributions.Normal(mean, std)
        pre_tanh_action = dist.rsample()
        tanh_action     = torch.tanh(pre_tanh_action)
        action          = self.rescale_fn(tanh_action)
        # tanh change-of-variables correction for log_prob.
        log_prob  = dist.log_prob(pre_tanh_action)
        log_prob -= torch.log((1 - tanh_action.pow(2)) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, value, tanh_action, mean, log_std

    def select_action(self, state):
        """
        Sample a stochastic action during training.
        Returns (action_np, logpa_np, value_np) — all needed by EpisodeBuffer.
        """
        action, log_prob, value, _, _, _ = self.full_pass(state)
        return (action.detach().cpu().numpy().squeeze(0),
                log_prob.detach().cpu().numpy().squeeze(),
                value.detach().cpu().numpy().squeeze())

    def select_greedy_action(self, state):
        """Select the deterministic (greedy) action using the mean directly."""
        mean, _, _ = self.forward(state)
        action = self.rescale_fn(torch.tanh(mean))
        return action.detach().cpu().numpy().squeeze(0)
