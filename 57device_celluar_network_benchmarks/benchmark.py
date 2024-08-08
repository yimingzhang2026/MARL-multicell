# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:38:55 2023

@author: zyimi
"""
from tqdm import tqdm
import argparse
from env import env
import random
import itertools
benchmark = 'LLQ2'
def distributed_ITLinQ(N, M, INR, SNR, C, eta, Pmax):
    active = np.zeros((N, M))  # Initialize active matrix
    for m in range(M):
        links = np.arange(N)  # Create an array of link indices
        np.random.shuffle(links)  # Shuffle the link indices
        active[links[0], m] = 1  # Initialize active(1) = 1 for the first link

        for j, link_j in enumerate(links[1:], start=1):
            Sj = [link_i for link_i in links[:j] if active[link_i, m] == 1]  # Set Sj as {i: i ≤ j and active(i) = 1}

            flagDj = 0
            flagSj = 0
            TEMP = C * (SNR[link_j, m] ** eta)  # Calculate MSNR for link j on channel m
            
            if all(INR[link_j, link_i, m] <= TEMP for link_i in Sj):  # Check if INR_ji ≤ MSNR at Dj, ∀i ∈ Sj
                flagDj = 1
    
            if all(INR[link_i, link_j, m] <= TEMP for link_i in Sj):  # Check if INR_ij ≤ MSNR at Sj, ∀i ∈ Sj
                flagSj = 1
            
            active[link_j, m] = flagDj * flagSj  # Set active(j) = flagDj * flagSj
    
    return active * Pmax

def fair_ITLinQ(N, M, INR, SNR, eta, eta_bar, C_value, C_bar, SNR_th, Pmax, queue):
    # Initialize the active matrix with zeros
    active = np.zeros((N, M))
    queue_length = [len(queue[i]) for i in range(N)]
    for m in range(M):
        # Create an array of link indices
        links = [i for i, length in enumerate(queue_length) if length > 0]
        # links = np.arange(N)
        # Shuffle the link indices
        np.random.shuffle(links)
        # Initialize active(1) = 1 for the first link in the shuffled list
        active[links[0], m] = 1

        for j, link_j in enumerate(links[1:], start=1):
            # Set Sj as {i: i ≤ j and active(i) = 1}
            Sj = [link_i for link_i in links[:j] if active[link_i, m] == 1]

            flagDj = 0
            flagSj = 0
            # Calculate MSNR for link j on channel m
            TEMP = C_value * (SNR[link_j, m] ** eta)
            
            # Check if INR_ji ≤ MSNR at Dj for all i in Sj
            if all(INR[link_j, link_i, m] <= TEMP for link_i in Sj):
                flagDj = 1
            
            # Check if SNR_j ≤ SNR_th
            if SNR[link_j, m] <= SNR_th:
                TEMP_Sj = C_value * (SNR[link_j, m] ** eta)
            else:
                TEMP_Sj = C_bar * (SNR[link_j, m] ** eta_bar)

            # Check if INR_ij ≤ MSNR at Sj for all i in Sj
            if all(INR[link_i, link_j, m] <= TEMP_Sj for link_i in Sj):
                flagSj = 1

            # Set active(j) = flagDj * flagSj
            active[link_j, m] = flagDj * flagSj
    
    # Return the active matrix scaled by Pmax
    return active * Pmax

def LLQ2(partial_obs, num_devices, M, Pmax):
    agent_action = np.zeros((num_devices, M))
    longest_q = max(partial_obs)
    subbands = [m for m in range(M)]
    indices = [i for i, val in enumerate(partial_obs) if val == longest_q and val !=0 and i < num_devices] # find the links with longest queue (cannot be empty, and can only operate links within the cell)
    if len(indices) == 1:
        agent_action[indices[0],:] = 1
    elif len(indices) > 1: #uniformly select 1 link with longest queue
        agent_action[random.choice(indices),:] = 1
    return agent_action * Pmax

def FP_algorithm_weighted(queue, N, H, Pmax, noise_var, M):
    queue_length = [len(queue[i]) for i in range(N)]
    sum_queue_length = np.sum(queue_length)
    actions = np.zeros((N, M))
    if sum_queue_length == 0:
        return actions
    else:
        weights = [x / sum_queue_length for x in queue_length]
    H_2 = H ** 2
    for m in range(M):
        f_new = 0
        gamma = np.zeros(N)
        y = np.zeros(N)
        p_init = Pmax * np.ones(N)
        # Initial power is just all transmitters transmit with full power
        p = np.array(p_init)
        # Take pow 2 of abs_H, no need to take it again and again
        H_2_sub = H_2[:,:,m]
        for i in range(N):
            tmp_1 = H_2_sub[i, i] * p[i]
            tmp_2 = np.matmul(H_2_sub[i, :], p) + noise_var
            # Initialize gamma
            gamma[i] = tmp_1 / (tmp_2 - tmp_1) #calculate SINR
        for iter in range(100):
            f_old = f_new
            for i in range(N):
                tmp_1 = H_2_sub[i, i] * p[i]
                tmp_2 = np.matmul(H_2_sub[i, :], p) + noise_var
                # Update y
                y[i] = np.sqrt(weights[i] * (1 + gamma[i]) * tmp_1) / (tmp_2)
                # Update gamma
                gamma[i] = tmp_1 / (tmp_2 - tmp_1)
    
            f_new = 0
            for i in range(N):
                # Update p
                p[i] = min (Pmax, (y[i] ** 2) * weights[i] * (1 + gamma[i]) * H_2_sub[i,i] / np.square(np.matmul(np.square(y), H_2_sub[:,i])))
            for i in range(N):
                # Get new result
                f_new = f_new + 2 * y[i] * np.sqrt(weights[i] * (1+gamma[i]) * H_2_sub[i,i] * p[i]) - (y[i] ** 2) * (np.matmul(H_2_sub[i, :], p)
                                                                                                                + noise_var)
            #Look for convergence
            if f_new - f_old <= 0.001:
                break

        actions[:,m] = p

    return actions


def WMMSE_algorithm_weighted(queue, N, H_, Pmax, noise_var, M, max_iterations=100, stopping_threshold=1e-3):
    actions = np.zeros((N, M))
    queue_length = [len(queue[i]) for i in range(N)]
    sum_queue_length = np.sum(queue_length)
    if sum_queue_length == 0:
        return actions
    else:
        weights = [x / sum_queue_length for x in queue_length]
    H_2 = H_ ** 2
    for m in range(M):
        H = H_2[:,:,m]
        #initilize
        f_new = 0
        v = np.sqrt(Pmax) * np.ones(N)#* np.random.rand(N) 
        u = np.zeros(N)
        w = np.zeros(N)
        # Compute u_0 = |H[i,i]|v_0 / (Σ(j=1 to N) |H[i,j]|^2(v_0)^2 + σ^2), ∀ i
        for i in range(N):
            interference_sum = 0
            for j in range(N):
                interference_sum += H[i, j] * np.square(v[j])
            u[i] = np.sqrt(H[i, i]) * v[i] / (interference_sum + noise_var)
        # Compute w_0 = 1 / (1 - u_0|H[i,i]|v_0), ∀ i
        for i in range(N):
            w[i] = 1 / (1 - u[i] * np.sqrt(H[i, i]) * v[i])

        for iter in range(100):
            f_old = f_new
            v_prev = v.copy() # Store the previous iteration's v
            # Update v
            for i in range(N):
                numerator = weights[i] * w[i] * u[i] * np.sqrt(H[i, i])
                denominator = 0
                for j in range(N):
                    denominator += weights[j] * w[j] * np.square(u[j]) * H[j, i]
                v[i] =  max(min(numerator / denominator, np.sqrt(Pmax)),0)
    
            # Update u
            for i in range(N):
                interference_sum = 0
                for j in range(N):
                    interference_sum += H[i, j] * np.square(v[j])
                u[i] = np.sqrt(H[i, i]) * v[i] / (interference_sum + noise_var)
    
            # Update w
            for i in range(N):
                w[i] = 1 / (1 - u[i] * np.sqrt(H[i, i]) * v[i] )
            
            
            
            #until Some stopping criteria is met
            f_new = 0
            # for i in range(N):
            #     term1 = 1 + (H[i, i] * v[i]) / (np.sum([H[i, j] * v[j] for j in range(N) if j != i]) + noise_var)
            #     f_new += weights[i] * np.log(term1)
    
            for i in range(N):
                ei = np.square(u[i] * np.sqrt(H[i, i]) * v[i] - 1) + np.sum([np.square(u[i] * np.sqrt(H[i, j]) * v[j]) for j in range(N) if j != i]) + noise_var * np.square(u[i])
                f_new += weights[i] * (w[i] * ei - np.log(w[i]))
            # Check the stopping criteria using the sum-rate metric
            # print(f_new)
            #Look for convergence
            if abs(f_new - f_old) <= stopping_threshold:
                break
            
        p_opt = np.square(v)
        actions[:,m] = p_opt
    return actions


def parse_args(parser):
    parser.add_argument('--scenario', type=str,
                     default='conflict_graph', help="Which scenario to run on")
    parser.add_argument('--K', type=int,
                        default=K, help="number of agents")
    
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="random seed",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=N,
        help="number of links in marl",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=3,
        help="number of subbands",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=5000,
        help="max duration of env in one episode, should be equal the args.episode_length parameter",
    )
    parser.add_argument(
        "--data_rates",
        type=list,
        default=[25] * N,
        help="the arrival rate of packets for agents, the length should be equal to the number of agents",
    )
    parser.add_argument(
        "--max_queue",
        type=int,
        default=50,
        help="the threshold of queue length to be considered as unstable",
    )
    parser.add_argument(
        "--random_deployment",
        type=bool,
        default=False,
        help="if set False, generate AP deployments in regular hexagonal",
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=True,
        help="if set False, generate AP deployments in regular hexagonal",
    )
                        
    all_args = parser.parse_args()  # Parse the arguments

    return all_args

#if False, AP are deployed in regular hexagon shape

K = 19
N = 57
parser = argparse.ArgumentParser()  # Create an empty parser
all_args = parse_args(parser)  # Parse the arguments using the parser

env = env(all_args)
#if test, set seed to make sure each time the env is the same
#print(env.poisson_process[1][:15])
import numpy as np
info_ep_list = []
progress_bar = tqdm(range(env.max_duration))
for i in range(1):
    obs = env.reset()
    done = False
    for step in progress_bar:
        #print(env.packets)
        #For each agent, the action is an L by M matrix, indicating the resource allocation of L devices on M subbands
        actions = np.zeros((env.N, env.M))
        if benchmark == 'LLQ2':
            for ap in range(env.K):
                agent_action = LLQ2(obs[ap], env.num_devices, env.M, env.Pmax) 
                for local_index in range(agent_action.shape[0]):
                    global_index = (local_index, ap)
                    if global_index not in env.local_to_global:
                        continue  # consider it as dummy link
                    else:
                        actions[env.local_to_global[global_index],:] = agent_action[local_index,:]
        elif benchmark == 'FP':
            actions = FP_algorithm_weighted(env.packets, env.N, env.H[-1], env.Pmax, env.noise_var, env.M)
            np.set_printoptions(precision=2)
            # print([len(env.packets[i]) for i in range(env.N)])
            # print(actions)
            # print(env.SINR)
        elif benchmark == 'WMMSE':
            actions = WMMSE_algorithm_weighted(env.packets, env.N, env.H[-1], env.Pmax, env.noise_var, env.M)
            np.set_printoptions(precision=2)
            # print([len(env.packets[i]) for i in range(env.N)])
            # print(actions)
            # print(env.SINR)
        elif benchmark == 'ITLQ':
            SNR, INR = env.get_INR_SNR()
            actions = distributed_ITLinQ(env.N, env.M, INR, SNR, 20, 0.6, env.Pmax)
        elif benchmark == 'FITLQ':
            SNR, INR = env.get_INR_SNR()
            if all_args.random_deployment:
                actions = fair_ITLinQ(env.N, env.M, INR, SNR, 0.7, 0.5, 20, 2, 5000, env.Pmax, env.packets) #good para for random deployment 20, 5
            else:
                actions = fair_ITLinQ(env.N, env.M, INR, SNR, 0.7, 0.5, 10, 0.1, 1200, env.Pmax, env.packets)  #good para for hexagonal deployment
        obs, reward, ternimated, info = env.step(actions)
        
        done = (any(ternimated) == True)

        if done or step == env.max_duration - 1:
            #print("episode_end!", "reward=", reward)
            info_ep = env.get_info_ep()
            info_ep_list.append(info_ep)
            print(info_ep['queue_length'])
            print(info_ep['over_all_ave_delay'])
            break

    
def save_data(file_path, data):
    np.savez(file_path, *data)
    print('data saved in {}'.format(file_path))
    
data = info_ep_list  # Get the data dictionary
save_data('./{}_ch{}_{}links_dr{}_rd{}.npz'.format(benchmark, all_args.M,all_args.N,all_args.data_rates[0],all_args.random_deployment), data) 
