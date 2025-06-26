import numpy as np
import multiprocessing as mp
import time
from copy import deepcopy


# ----- Utils -----
def split_rho(rho):
        "Separate real and imag parts of DM, flatten"
        r, im = np.real(rho).reshape(-1), np.imag(rho).reshape(-1)
        return np.stack((r, im), axis=1).reshape(-1)


# ----- Training -----
def instant_reward_training(n_its, env, agent, prnt_ep=True, prnt_it=1, avg=100,
                            chkpt=100):
    reward_history = []  # Episode rewards
    reward_rolling_avg = []  # Rolling avg of rewards
    best_r = 0  # Best reward yet
    last_saved = 0  # Last episode models were saved at
    for i in range(n_its):
        env.reset()
        done = False

        t1 = time.time()  # Start episode timer
        while not done:
            obs = np.append(
                 split_rho(env.rho),
                 env.T_tot / env.T_lim).reshape(1, -1)  # Current state
            act = agent.choose_action(obs)  # Choose action
            act_s = env.scale_action(act)  # Scale to action space
            env.apply_kick(act_s[0], act_s[1], act_s[2])  # Compute transition
            obs_ = np.append(
                 split_rho(env.rho),
                 env.T_tot/env.T_lim).reshape(1, -1)  # New state
            rew = env.compute_reward()
            if len(env.actions) == env.N_pulses:
                done = True
            agent.remember(obs, act, rew, obs_, done)
            agent.learn()
        t2 = time.time() - t1  # End episode timer

        # Debug Info
        reward_history.append(np.sum(env.rewards))  # Final sum of rewards
        avg_reward = np.mean(reward_history[-1 * avg:])  # Avg over last `avg` rewards
        reward_rolling_avg.append(avg_reward)
        if avg_reward > best_r:
            if abs(i - last_saved) <= 1:
                 time.sleep(2)  # Pause to ensure previous files finish writing
            last_saved = i  # Update when saved
            best_r = avg_reward  # New best avg
            agent.save_models()
        
        # Debug
        if prnt_ep and i%prnt_it==0:
            print(f"    Episode: {i} | Reward: {reward_history[-1] : .3f} " + \
                f"| Average Reward: {avg_reward : .3f} | Time: {t2 : .2f} s")
        
        # Checkpoint
        if i%chkpt==0:
            # agent.memory.save_buffer()
            ax = [j + 1 for j in range(i)]
            np.savez(agent.chkpt_dir + "learning_curve", allow_pickle=False,
                    rewards=reward_history, roll_avg=reward_rolling_avg, it_ax=ax)

    if prnt_ep:
         print(f"Model Last Saved at Episode {last_saved}")
    return reward_history, reward_rolling_avg


# ----- Policy Evaluation -----
def sample_policy_traj(env, agent):
    # Obtain samples and mean trajectory
    env.reset()
    done = False
    while not done:
        obs = np.append(split_rho(env.rho), env.T_tot).reshape(1, -1)  # Current state
        act = agent.choose_action(obs)
        act_s = env.scale_action(act)  # Scale to action space
        env.apply_kick(act_s[0], act_s[1], act_s[2])  # Compute transition

        if len(env.actions) == env.N_pulses:
                done = True

    # Output
    Omegas = np.array([a[0] for a in env.actions])
    phis = np.array([a[1] for a in env.actions])
    deltas = np.array([a[2] for a in env.actions])
    env.reset()
    return (Omegas, phis, deltas)


def full_sim_mp_wrapper(env, act, t_ax, t_qax):
    "Simulate trajectory from actions, wrapper for multiprocessing"
    env.reset()
    Om, phi, de = act
    rhos, qfis, ts = env.full_sim(Om, phi, de, t_ax, t_qax)
    env.reset()
    return rhos, qfis, ts


def policy_samples(env, agent, t_ax, t_qax, samples=50, nprocs=1):
    "Sample trajectories from agent policy and simulate"
    env = deepcopy(env)  # Need deepcopy for MP
    # Obtain samples
    acts = []
    for i in range(samples):
         acts.append(sample_policy_traj(env, agent))

    # Simulate trajectories w/ MP
    results = []
    with mp.Pool(nprocs) as pool:
        for a in acts:
            results.append(pool.apply_async(full_sim_mp_wrapper, (env, a, t_ax, t_qax)))
        pool.close()
        pool.join()

    results = [res.get() for res in results]
    rhos = [r[0] for r in results]
    qfis = [r[1] for r in results]
    ts = results[0][2]
    return acts, rhos, qfis, ts
