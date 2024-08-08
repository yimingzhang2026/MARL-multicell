# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:38:55 2023

@author: zyimi
"""
import argparse
from env import env
import random
import copy
import math

def gms_multi_subband(env):
    actions =np.zeros((env.N, env.M))
    conflicts = copy.copy(env.conflicts)
    queue_length = [len(env.packets[i]) for i in range(env.N)]
    
    participate_links = [i for i, q in enumerate(queue_length) if q != 0 ]
    # print(participate_links)
    # Calculate weights for each link
    weights = {i: math.log(10 * queue_length[i]) for i in participate_links}
    
    while participate_links:
        # Find the links with the maximum weight
        max_weight = max(weights.values())
        max_weight_links = [link for link, weight in weights.items() if weight == max_weight]
        
        if len(max_weight_links) > 1:
            max_weight_link = random.choice(max_weight_links)
        else:
            max_weight_link = max_weight_links[0]
            
        actions[max_weight_link][:] = 1
        # Remove the scheduled link and its conflicting links from participate_links
        participate_links.remove(max_weight_link)
        weights.pop(max_weight_link, None)
        for conflict in conflicts[max_weight_link]:
            if conflict in participate_links:
                participate_links.remove(conflict)
                weights.pop(conflict, None)
    return actions

# def gms_multi_subband(env):
#     actions =np.zeros((env.N, env.M))
#     conflicts = copy.copy(env.conflicts)
#     queue_length = [len(env.packets[i]) for i in range(env.N)]
    
#     for m in range(env.M):
#         participate_links = [i for i, q in enumerate(queue_length) if q != 0 ]
#         # print(participate_links)
#         # Calculate weights for each link
#         weights = {i: math.log(10 * queue_length[i]) for i in participate_links}
        
#         while participate_links:
#             # Find the links with the maximum weight
#             max_weight = max(weights.values())
#             max_weight_links = [link for link, weight in weights.items() if weight == max_weight]
            
#             if len(max_weight_links) > 1:
#                 max_weight_link = random.choice(max_weight_links)
#             else:
#                 max_weight_link = max_weight_links[0]
                
#             actions[max_weight_link][m] = 1
#             # Remove the scheduled link and its conflicting links from participate_links
#             participate_links.remove(max_weight_link)
#             weights.pop(max_weight_link, None)
#             for conflict in conflicts[max_weight_link]:
#                 if conflict in participate_links:
#                     participate_links.remove(conflict)
#                     weights.pop(conflict, None)
            
#             #update queue length
#             for link in range(len(actions)):
#                 if actions[link][m] == 1:
#                     queue_length[link] -= 1
#                     queue_length[link] = max(queue_length[link], 0)
                
#     return actions


    


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
        default=[0.5] * N,
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
W = 2*N
parser = argparse.ArgumentParser()  # Create an empty parser
all_args = parse_args(parser)  # Parse the arguments using the parser

env = env(all_args)
#if test, set seed to make sure each time the env is the same
obs = env.reset()
#print(env.poisson_process[1][:15])
import numpy as np
info_ep_list = []
actions = np.zeros((env.N,env.M))

for i in range(1):
    done = False
    for step in range(env.max_duration):
        # print(f"Step {step + 1}")
        #print(env.packets)
        actions = gms_multi_subband(env)
        ap_actions = [np.zeros((env.num_devices,env.M)) for _ in range(env.K)]
        for global_index, link_action in enumerate(actions):
            local_index, ap = env.global_to_local[global_index]
            ap_actions[ap][local_index,:] = link_action
        obs, reward, ternimated, info = env.step(ap_actions)
        
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
save_data('./gms_ch{}_{}nodes_dr{}.npz'.format(all_args.M,all_args.N,all_args.data_rates[0]), data) 
