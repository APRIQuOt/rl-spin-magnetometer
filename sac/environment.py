import numpy as np

# Julia before torch
from juliacall import Main as jl
jl.seval("using SensorUtils")


class NPulse_Environment():
    def __init__(self, spin, omega, Ls, gamma_s, rho0, T_Omega, T_lim, N_pulses, 
                 bounds, chkpt_dir):
        # Unpack args
        self.spin = spin  # Spin manifold
        self.omega = omega  # Precession freq. [rad/s]
        self.Ls = Ls  # Jump operators
        self.gamma_s = gamma_s  # Decay rates [rad/s]
        self.rho0 = rho0  # Initial DM
        self.T_Omega = T_Omega  # Pulse duration [s]
        self.T_lim = T_lim  # Time limit on problem [s]
        self.N_pulses = N_pulses  # Total number of pulses being applied each episode
        self.bounds = bounds  # Action space bounds
        self.chkpt_dir = chkpt_dir  # Directory for saving parameters

        # Computed
        self.dims = int(2 * self.spin + 1)  # Number of states
        self.a_maxs = np.array([b[0] for b in self.bounds])  # Max action scaling
        self.a_mins = np.array([b[1] for b in self.bounds])  # Min action scaling

        # Training attributes
        self.observation_shape = (int(2 * (self.dims**2)) + 1,)  # obs space shape
        self.action_shape = (3,)  # action space shape
    
    def scale_action(self, action):
        "Scale actions from [-1, 1] to action space"
        action_s = 0.5 * (action + 1)  # Scale to [0, 1]
        span = self.a_maxs - self.a_mins
        action_s = (action_s * span + self.a_mins)  # Scale
        return action_s

    def apply_kick(self, Omega, phi, delta):
        " Apply a single kick to stored state `self.rho`, and append to `self.actions`"
        # As arrays for type-stable performance
        Omega = np.array([Omega])
        phi = np.array([phi])
        delta = np.array([delta])

        self.rho = np.array(
            jl.apply_pulses(self.rho, np.array([self.T_Omega]), self.spin, self.omega,
                            Omega, phi, delta, np.array([0.]), self.T_Omega,
                            self.Ls, self.gamma_s
                            )[-1])
        
        self.T_tot += self.T_Omega  # Increase total time
        self.actions.append([Omega, phi, delta])

    def compute_reward(self):
        "Compute reward for stored trajectory"
        Omegas = np.array([a[0] for a in self.actions])
        phis = np.array([a[1] for a in self.actions])
        deltas = np.array([a[2] for a in self.actions])
        ts = np.array([i * self.T_Omega for i in range(len(self.actions))])

        QFI = jl.qfi_pulsed(
            self.rho0, np.array([self.T_tot]), self.spin, self.omega, Omegas, phis,
            deltas, ts, self.T_Omega, self.Ls, self.gamma_s
            )[-1]
        
        if len(self.rewards) == 0:
            rew = QFI
        else:
            rew = QFI - np.sum(self.rewards)
        self.rewards.append(rew)
        return rew
    
    def full_sim(self, Omegas, phis, deltas, t_ax, t_qax):
        "Simulate stored trajectory in full, usually for plotting/evaluation"
        if not isinstance(Omegas, np.ndarray):
            Omegas = np.array(Omegas)
        if not isinstance(phis, np.ndarray):
            phis = np.array(phis)
        if not isinstance(deltas, np.ndarray):
            deltas = np.array(deltas)
        ts = np.array([i * self.T_Omega for i in range(len(Omegas))])  # Kick times

        rhos = jl.apply_pulses(self.rho0, t_ax, self.spin, self.omega, Omegas, phis,
                               deltas, ts, self.T_Omega, self.Ls, self.gamma_s)
        rhos = np.array([r for r in rhos])  # As np.ndarray
        qfis = np.array([
            jl.qfi_pulsed(self.rho0, t_qax, self.spin, self.omega, Omegas, phis, deltas,
                          ts, self.T_Omega, self.Ls, self.gamma_s)
        ])
        return rhos, qfis, ts
    
    def reset(self):
        "Reset stored trajectories"
        self.rho = self.rho0  # Reset to initial state
        self.T_tot = 0  # Reset elapsed time
        self.actions = []  # Empty actions
        self.rewards = []  # Empty rewards

    def save_params(self):
        "Store __init__ parameters to `chkpt_dir`"
        env_params = {
            "spin" : self.spin,
            "omega" : self.omega,
            "N_pulses" : self.N_pulses,
            "Ls" : self.Ls,
            "gamma_s" : self.gamma_s,
            "rho0" : self.rho0,
            "T_Omega" : self.T_Omega,
            "T_lim" : self.T_lim,
            "bounds" : self.bounds
        }
        np.save(self.chkpt_dir + "env_params", env_params)
