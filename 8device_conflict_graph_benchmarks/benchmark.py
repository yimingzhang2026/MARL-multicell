# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:38:55 2023

@author: zyimi
"""
import argparse
from env import env
import random

# def benchmark_simple1(partial_obs, num_devices, M):
#     agent_action = np.zeros((num_devices, M))
#     for m in range(M):
#         longest_q = max(partial_obs)
#         indices = [i for i, val in enumerate(partial_obs) if val == longest_q and val !=0 and i < num_devices] # find the links with longest queue (cannot be empty, and can only operate links within the cell)
#         if len(indices) == 0: # all links are empty
#             continue
#         elif len(indices) == 1:
#             scheduled_link = indices[0]
#         elif len(indices) > 1: #uniformly select 1 link with longest queue
#             scheduled_link = random.choice(indices)
#         agent_action[scheduled_link,m] = 1
#         partial_obs[scheduled_link] -= 1
#         partial_obs[scheduled_link] = max(partial_obs[scheduled_link], 0)
#     return agent_action
    
def LLQ(partial_obs, num_devices, M):
    agent_action = np.zeros((num_devices, M))
    longest_q = max(partial_obs)
    subbands = [m for m in range(M)]
    indices = [i for i, val in enumerate(partial_obs) if val == longest_q and val !=0 and i < num_devices] # find the links with longest queue (cannot be empty, and can only operate links within the cell)
    if len(indices) == 1:
        agent_action[indices[0],:] = 1
    elif len(indices) > 1: #uniformly select 1 link with longest queue
        agent_action[random.choice(indices),:] = 1
    return agent_action

def parse_args(parser):
    parser.add_argument('--scenario', type=str,
                     default='conflict_graph', help="Which scenario to run on")
    parser.add_argument('--K', type=int,
                        default=K, help="number of agents")
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
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
        default=[0.3] * N,
        help="the arrival rate of packets for agents, the length should be equal to the number of agents",
    )
    parser.add_argument(
        "--max_queue",
        type=int,
        default=50,
        help="the threshold of queue length to be considered as unstable",
    )
                        
    all_args = parser.parse_args()  # Parse the arguments

    return all_args


K = 4
N = 8
parser = argparse.ArgumentParser()  # Create an empty parser
all_args = parse_args(parser)  # Parse the arguments using the parser

env = env(all_args)
#if test, set seed to make sure each time the env is the same
obs = env.reset()
#print(env.poisson_process[1][:15])
import numpy as np
info_ep_list = []

for i in range(1):
    done = False
    for step in range(env.max_duration):
        # print(f"Step {step + 1}")
        #print(env.packets)
        #For each agent, the action is an L by M matrix, indicating the resource allocation of L devices on M subbands
        actions = [np.zeros((len(env.service_pool[ap]), env.M)) for ap in range(env.K)]
        for ap in range(env.K):
            actions[ap] = LLQ(obs[ap], len(env.service_pool[ap]), env.M)
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
save_data('./benchmark_ch{}_{}nodes_dr{}.npz'.format(all_args.M,all_args.N,all_args.data_rates[0]), data) 
