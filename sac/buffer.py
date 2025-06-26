import os
import numpy as np


class ReplayBuffer():
    def __init__(self, mem_size, input_shape, n_actions, chkpt_dir, name="buffer",
                 print_out=True):
        self.mem_size = mem_size  # Max size of buffer
        self.input_shape = input_shape  # Input size
        self.n_actions = n_actions  # Size of action space
        self.chkpt_dir = chkpt_dir  # Directory to save/load
        self.print_out = print_out  # Whether to print output or not

        self.chkpt_file = os.path.join(self.chkpt_dir, name)

        # Empty buffers
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.mem_cntr = 0  # Set counter to 0

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
    
    def reset(self):
        self.state_memory = np.zeros((self.mem_size, *self.input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *self.input_shape))
        self.action_memory = np.zeros((self.mem_size, self.n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def save_buffer(self):
        if self.print_out: print("... saving transitions in buffer...")
        np.savez(self.chkpt_file, s=self.state_memory, s_=self.new_state_memory,
                 a=self.action_memory, r=self.reward_memory, d=self.terminal_memory,
                 m=self.mem_cntr)

    def load_buffer(self):
        if self.print_out: print("... loading buffer from file...")
        with np.load(self.chkpt_file + ".npz") as data:
            self.state_memory = data["s"]
            self.new_state_memory = data["s_"]
            self.action_memory = data["a"]
            self.reward_memory = data["r"]
            self.terminal_memory = data["d"]
            self.mem_cntr = data["m"]
