import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, chkpt_dir, device,
                 fc_dims=[256, 256], name="critic", ):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims  # Input size
        self.n_actions = n_actions  # Size of action space
        self.chkpt_dir = chkpt_dir  # Directory to save/load
        self.device = device  # Device to store tensors on

        self.chkpt_file = os.path.join(self.chkpt_dir, name + ".pt")

        # NN
        self.layers = nn.ModuleList()  # Module list of layers + activations
        activation = nn.ReLU()
        in_size = self.input_dims + n_actions  # Input size
        for size in fc_dims:
            self.layers.append(nn.Linear(in_size, size))  # Add layer
            in_size = size  # For the next layer
            self.layers.append(activation)  # Add activation
        self.layers.append(nn.Linear(in_size, 1))  # Output

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(self.device)  # Send to device

    def forward(self, state, action):
        input_data = torch.cat([state, action], dim=1)
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file, weights_only=False))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, chkpt_dir, device,
                 fc_dims=[256, 256], name="value", ):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims  # Input size
        self.chkpt_dir = chkpt_dir  # Directory to save/load
        self.device = device  # Device to store tensors on
        
        self.chkpt_file = os.path.join(self.chkpt_dir, name + ".pt")

        # NN
        self.layers = nn.ModuleList()  # Module list of layers + activations
        activation = nn.ReLU()
        in_size = self.input_dims  # Input size
        for size in fc_dims:
            self.layers.append(nn.Linear(in_size, size))  # Add layer
            in_size = size  # For the next layer
            self.layers.append(activation)  # Add activation
        self.layers.append(nn.Linear(in_size, 1))  # Output

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(self.device)  # Send to device

    def forward(self, state):
        input_data = state
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file, weights_only=False))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, chkpt_dir, device,
                 fc_dims=[256, 256], n_actions=3, name="actor"):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims  # Input size
        self.chkpt_dir = chkpt_dir  # Directory to save/load
        self.device = device  # Device to store tensors on
        self.n_actions = n_actions  # Size of action space
        
        self.chkpt_file = os.path.join(self.chkpt_dir, name + ".pt")
        self.reparam_noise = 1e-6  # For avoiding log(0)

        # NN
        self.layers = nn.ModuleList()  # Module list of layers + activations
        activation = nn.ReLU()
        in_size = self.input_dims  # Input size
        for size in fc_dims:
            self.layers.append(nn.Linear(in_size, size))  # Add layer
            in_size = size  # For the next layer
            self.layers.append(activation)  # Add activation
        self.mu = nn.Linear(in_size, n_actions)  # Means
        self.sigma = nn.Linear(in_size, n_actions)  # Standard devs

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(self.device)  # Send to device

    def forward(self, state):
        input_data = state
        for layer in self.layers:
            input_data = layer(input_data)
        mu = self.mu(input_data)  # Means
        sigma = self.sigma(input_data)  # Std. devs.
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)  # Restrict to >0
        return mu, sigma
    
    def sample_policy(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            action_samp = probabilities.rsample()
        else:
            action_samp = probabilities.sample()

        action = torch.tanh(action_samp)  # Bound action space by tanh()
        log_probs = probabilities.log_prob(action_samp)  # From sampled action
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)  # Modification
        log_probs = log_probs.sum(1, keepdim=True)  # Sum over actions

        return action, log_probs
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file, weights_only=False))
