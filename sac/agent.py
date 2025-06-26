import numpy as np
import torch
import torch.nn.functional as fun
from sac.buffer import ReplayBuffer
from sac.networks import ActorNetwork, CriticNetwork, ValueNetwork


class Agent:
    def __init__(self, input_dims, n_actions, reward_scale, chkpt_dir,
                 alpha=0.0003, beta=0.0003, gamma=0.99, tau=0.005, batch_size=256,
                 mem_size=100000, c_dims=[256, 256], a_dims=[256, 256],
                 v_dims=[256, 256], print_out=True, device=None):
        self.input_dims = input_dims  # Input size
        self.n_actions = n_actions  # Size of action space
        self.scale = reward_scale  # Scaling on reward vs. entropy
        self.chkpt_dir = chkpt_dir  # Directory to save/load
        self.gamma = gamma  # Discounted reward HP
        self.tau = tau  # Target value NN HP
        self.batch_size = batch_size
        self.mem_size = mem_size  # Max size of buffer
        self.a_dims = a_dims  # Actor NN neurons
        self.c_dims = c_dims  # Critic NN neurons
        self.v_dims = v_dims  # Value NN neurons
        self.print_out = print_out  # Whether to print outputs or not

        # Setup buffer
        self.memory = ReplayBuffer(self.mem_size, (input_dims,), n_actions,
                                   self.chkpt_dir, print_out=self.print_out)

        # Device
        if device is None:
            if torch.cuda.is_available():
                dev_string = "... using CUDA device..."
                self.device = torch.device("cuda")  # NVIDIA GPU
            elif torch.backends.mps.is_available():
                dev_string = "... using Apple device..."
                self.device = torch.device("mps")  # Apple GPU
            else:
                dev_string = "... using CPU..."
                self.device = torch.device("cpu")
            
            if self.print_out: print(dev_string)
        else:
            self.device = device

        # NNs
        self.actor = ActorNetwork(alpha, input_dims, self.chkpt_dir,self.device,
                                  n_actions=self.n_actions, fc_dims=a_dims, name="actor")
        self.critic_1 = CriticNetwork(beta, input_dims, self.n_actions, self.chkpt_dir,
                                      self.device, fc_dims=c_dims, name="critic_1")
        self.critic_2 = CriticNetwork(beta, input_dims, self.n_actions, self.chkpt_dir,
                                      self.device, fc_dims=c_dims, name="critic_2")
        self.value = ValueNetwork(beta, input_dims, self.chkpt_dir, self.device,
                                  fc_dims=v_dims, name="value")
        self.target_value = ValueNetwork(beta, input_dims, self.chkpt_dir, self.device,
                                  fc_dims=v_dims, name="target_value")

        self.update_network_parameters(tau=1)  # target_value NN as soft copy of value NN

    def choose_action(self, observation):
        "Sample actions from policy, no gradients"
        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        actions, _ = self.actor.sample_policy(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, new_state, done):
        "Store transition in buffer"
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        "Update rule for target value NN, as soft copy of value NN"
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                (1 - tau) * target_value_state_dict[name].clone()
            
        self.target_value.load_state_dict(value_state_dict)
    
    def learn(self):
        "Update all NNs"
        if self.memory.mem_cntr < self.batch_size:
            "Not enough experience yet, skip learning"
            return
        # Recall experiences
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        # As tensors
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        state_ = torch.tensor(state_, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)
        
        value = self.value(state).view(-1)  # Re-evaluated state values
        value_ = self.target_value(state_).view(-1)  # Values of next state
        value_[done] = 0.0  # Set to 0 if terminal

        # Update value NN
        critic_value, log_probs = self.evaluate_policy(state, reparameterize=False)
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * fun.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Update actor NN
        critic_value, log_probs = self.evaluate_policy(state, reparameterize=True)
        self.actor.optimizer.zero_grad()
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Update critic NNs
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * fun.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * fun.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Update target value NN
        self.update_network_parameters()

    def evaluate_policy(self, state, reparameterize=True):
        "Sample new actions from current policy, criticize"
        # Draw actions from current policy
        action, log_probs = \
            self.actor.sample_policy(state, reparameterize=reparameterize)
        log_probs = log_probs.view(-1)

        # Q-function for new actions
        q1_new_policy = self.critic_1.forward(state, action)
        q2_new_policy = self.critic_2.forward(state, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy)  # Min Double Q
        critic_value = critic_value.view(-1)

        return critic_value, log_probs

    def save_models(self):
        if self.print_out: print("... saving models ...")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_value.save_checkpoint()

    def save_agent_params(self):
        agent_params = {
            "scale" : self.scale,
            "gamma" : self.gamma,
            "tau" : self.tau,
            "batch_size" : self.batch_size,
            "mem_size" : self.mem_size,
            "a_dims" : self.a_dims,
            "c_dims" : self.c_dims,
            "v_dims" : self.v_dims
        }
        np.save(self.chkpt_dir + "agent_params", agent_params)

    def load_models(self):
        if self.print_out: print("... loading models ...")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_value.load_checkpoint()

    def reset_buffer(self):
        if self.print_out: print("... reseting replay buffer ...")
        self.memory.reset()
