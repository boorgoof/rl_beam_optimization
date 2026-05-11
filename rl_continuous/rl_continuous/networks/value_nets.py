import torch
import torch.nn as nn
import torch.nn.functional as F


class FCQV(nn.Module):
    """
    FCQV stands for Fully Connected Q-Value network.

    @param obs_dim: dimension of the state space
    @param act_dim: dimension of the action space
    @param hidden_dims: tuple specifying the number of neurons in each hidden layer (default: (256, 256), that is two hidden layers with 256 neurons each)
    @param activation_fc: activation function used by the network (default: F.relu) 
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_dims=(256, 256),
                 activation_fc=F.relu):
        
        super(FCQV, self).__init__()

        # activation function
        self.activation_fc = activation_fc
        # Input layer: maps the input (of size obs_dim + act_dim) to the first hidden layer size
        self.input_layer = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        # Hidden layers: connect consecutive hidden dimensions.
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        # Output layer maps the last hidden layer to a single scalar output (the Q-value)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def _format(self, state, action):
        """Function to be sure that state and action are tensors of the expected shape i.e. (batch_size, feature_obs_dim) and (batch_size, act_dim) respectively."""
        x, u = state, action 
        if not isinstance(x, torch.Tensor): # expected a tensor with torch.Size([batch_size, obs_dim]). i.e (1, 3) for a single state with obs_dim=3. (1, 1) for a single action with act_dim=1.
            x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32)
            if u.dim() == 1:
                u = u.unsqueeze(0)
        return x, u

    def forward(self, state, action):
        """
        It takes the pair (state, action) as input and outputs a scalar Q(s,a).

        @param state: the state input of shape (batch_size, obs_dim) after formatting
        @param action: the action input of shape (batch_size, act_dim) after formatting
        @return: the Q-value estimate for the given state-action pair, of shape (batch_size, 1)
        """
        # check and format state and action
        x, u = self._format(state, action)
        # concatenate state and action in the feature dimension (dim=1), i.e. (1,3) + (1,1) = (1,4)
        x = torch.cat((x, u), dim=1)
        # pass through the network: input layer, hidden layers, and output layer, applying activation function after each layer except the output.
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)


class FCV(nn.Module):
    """
    FCV stands for Fully Connected Value network. Takes a state and outputs a scalar V(s).

    @param obs_dim: dimension of the state space
    @param hidden_dims: tuple specifying the number of neurons in each hidden layer (default: (256, 256), that is two hidden layers with 256 neurons each)
    @param activation_fc: activation function used by the network (default: F.relu) 
    """
    def __init__(self,
                 obs_dim,
                 hidden_dims=(256, 256),
                 activation_fc=F.relu):
        
        super(FCV, self).__init__()

        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def _format(self, state):
        """Function to be sure that state is a tensor of the expected shape i.e. (batch_size, feature_obs_dim)."""
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        """
        It takes the state as input and outputs a scalar V(s).

        @param state: the state input of shape (batch_size, obs_dim) after formatting
        @return: the value estimate for the given state, of shape (batch_size, 1)
        """

        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)


class FCTQV(nn.Module):
    """
    FCTQV stands for Fully Connected Twin Q-Value network.
    It contains two independent Q-value  (a and b) that share the same input. Using the minimum of the two reduces overestimation bias.
    @param obs_dim: dimension of the state space
    @param act_dim: dimension of the action space
    @param hidden_dims: tuple specifying the number of neurons in each hidden layer (default: (256, 256), that is two hidden layers with 256 neurons each)
    @param activation_fc: activation function used by the network (default: F.relu)
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_dims=(256, 256),
                 activation_fc=F.relu):
        super(FCTQV, self).__init__()

        self.activation_fc = activation_fc

        # network A
        self.input_layer_a = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        self.hidden_layers_a = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers_a.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer_a = nn.Linear(hidden_dims[-1], 1)

        # network B, independent from A but with the same architecture.
        self.input_layer_b = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        self.hidden_layers_b = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers_b.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer_b = nn.Linear(hidden_dims[-1], 1)

    def _format(self, state, action):
        """Function to be sure that state and action are tensors of the expected shape i,e. (batch_size, feature_obs_dim) and (batch_size, act_dim) respectively."""
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32)
            if u.dim() == 1:
                u = u.unsqueeze(0)
        return x, u

    def forward(self, state, action):
        """Forward pass through the network.

        @param state: the state input of shape (batch_size, obs_dim) after formatting
        @param action: the action input of shape (batch_size, act_dim) after formatting
        @return: a tuple (Q_a, Q_b) where each is the Q-value
        """

        x, u = self._format(state, action)
        xu = torch.cat((x, u), dim=1)  # concatenate state and action in the feature dimension (dim=1), i.e. (1,3) + (1,1) = (1,4)
       
        # Stream A.
        xa = self.activation_fc(self.input_layer_a(xu))
        for hidden_layer_a in self.hidden_layers_a:
            xa = self.activation_fc(hidden_layer_a(xa))
        qa = self.output_layer_a(xa)
        
        # Stream B.
        xb = self.activation_fc(self.input_layer_b(xu))
        for hidden_layer_b in self.hidden_layers_b:
            xb = self.activation_fc(hidden_layer_b(xb))
        qb = self.output_layer_b(xb)
        return qa, qb

    def Qa(self, state, action):
        """
        Qa returns only the first stream's Q-value estimate. It is used when calculating the targets for the policy updates.
        """
        # it is a normal forward pass through stream A, but we ignore stream B.
        x, u = self._format(state, action)
        xu = torch.cat((x, u), dim=1)
        xa = self.activation_fc(self.input_layer_a(xu))
        for hidden_layer_a in self.hidden_layers_a:
            xa = self.activation_fc(hidden_layer_a(xa))
        return self.output_layer_a(xa)


