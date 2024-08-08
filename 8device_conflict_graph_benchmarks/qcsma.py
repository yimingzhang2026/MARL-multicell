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

def qcsma_multi_subband(env,previous_actions,W):
    actions = copy.copy(previous_actions)
    back_off_times = np.zeros(len(actions))
    conflicts = copy.copy(env.conflicts)
    queue_length = [len(env.packets[i]) for i in range(env.N)]
    # Step 1: Calculate wl for each link
    w = [math.log(10 * q) if q!=0 else -float('inf') for q in queue_length]
    for m in range(env.M):
        decision_set = set()
        
        #w = [math.log(1 + q) for q in queue_length]
        w = [math.log(10 * q) if q!=0 else -float('inf') for q in queue_length]
        # Step 2: Choose random backoff time in [0, W-1]
        for i in range(env.N):
          #back_off_times[i] = random.randint(0, W - 1)
          back_off_times[i] = random.randint(0, W - 1) if queue_length[i] > 0 else W + 10
         
        # Step 3: Check for control message collisions
        intents = {i : set() for i in range(W + 1)}
        for mini_slot in range(W):
            # Identify links transmitting an INTENT message in this mini_slot.
            transmitting_links = [i for i, back_off_time in enumerate(back_off_times) if back_off_time == mini_slot]
            intents[mini_slot + 1] = copy.copy(intents[mini_slot])
            # Check for collisions in this mini_slot and update intents and decision_set.
            for link in transmitting_links:
                if any(conflict in intents[mini_slot] for conflict in conflicts[link]):
                    continue  # If collision, skip this link.
                intents[mini_slot + 1].add(link)  # Add link to intents.
        intent_transmitting_links = intents[W]
        decision_set = []
        #print(decision_set)
        for link in intent_transmitting_links:
            has_conflicts_in_list = any(conflict in decision_set for conflict in conflicts[link])
            if not has_conflicts_in_list:
                decision_set.append(link)
        #print(decision_set)
        
        # Step 4: Process links in decision set
        for link in decision_set:
            ewl = math.exp(w[link])
            p = ewl / (1 + ewl)
            # If no conflicts in decision_set were active in the previous slot, decide state with probability p.
            if all(previous_actions[conflict][m] == 0 for conflict in env.conflicts[link]):
                actions[link][m] = 1 if random.random() < p else 0
            else:
                actions[link][m]  = 0  
                
        for link in range(len(actions)):
            if queue_length[link] == 0:
                actions[link][m]  = 0 

                
    return actions


    


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
        actions = qcsma_multi_subband(env,actions,W)
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
save_data('./qcsma_ch{}_{}nodes_dr{}.npz'.format(all_args.M,all_args.N,all_args.data_rates[0]), data) 
