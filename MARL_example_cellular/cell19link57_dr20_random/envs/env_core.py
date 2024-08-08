import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random
import copy
import math
from scipy import special
import collections
from envs.env_backend import *

non_zero_divison = 1e-7
max_spectual_eff = np.log2(1.0+1000)
class EnvCore(object):
    """
    # env agent def
    """

    def __init__(self,args):
        ##multi-agent env
        self.agent_num = self.K = args.K  
        self.N = args.N
        self.M = args.M
        self.max_duration = args.episode_length
        self.data_rates = args.data_rates
        self.seed = args.seed
        self.test = args.test
        ##cellular setting
        self.unstable_th = args.max_queue
        self.random_deployment = args.random_deployment #if False, AP are deployed in regular hexagon shape
        self.R = 500 # the radius of base station
        self.Pmax_dBm = 23.0
        self.n0_dBm = -114.0
        self.rayleigh_var = 1.0
        # self.shadowing_dev = 6.0
        self.f_d = 10.0
        self.packet_size = 5e5
        self.bw = 10e6 #Hz
        self.SINR = np.zeros((self.N, self.M))
        self.spec_eff = np.zeros((self.N, self.M))
        self.T = 0.02
        self.Pmax = np.power(10.0,(self.Pmax_dBm - 30)/10)
        self.correlation = special.j0(2.0*np.pi*self.f_d*self.T)
        self.noise_var = np.power(10.0,(self.n0_dBm - 30)/10)
        self.H = collections.deque([],2)
        self.throughput = np.zeros(self.N)
        #add diff power level
        self.action_dim_power = 6
        self.Pmax_dBw = self.Pmax_dBm - 30
        Pmin_dB = 0.0-30
        self.powers = np.zeros(self.action_dim_power)
        strategy_translation_dB_step = (self.Pmax_dBw-Pmin_dB)/(self.action_dim_power-2)
        for i in range(1,self.action_dim_power-1):
            self.powers[i] = np.power(10.0,((Pmin_dB+(i-1)*strategy_translation_dB_step))/10)     
        self.powers[-1] = self.Pmax
        #topology info
        if self.random_deployment:
            self.TX_loc, self.RX_loc, self.cell_mapping, self.service_pool = generate_deployment_random(self.R, self.K, self.N, self.seed)
        else:
            self.TX_loc, self.RX_loc, self.cell_mapping, self.service_pool = generate_deployment_hexa(self.R, self.K, self.N, self.seed)
        print(self.service_pool)
        self.num_devices = max([len(x) for x in self.service_pool])
        print(self.num_devices)
        self.global_to_local, self.local_to_global = global_local_index_mapping(self.service_pool)
        ## Plot setup
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.scatter(self.RX_loc[0], self.RX_loc[1], color='blue', label='Devices')
        # ax.scatter(self.TX_loc[0], self.TX_loc[1], color='red', label='APs')
        # # Label each device and AP
        # for i, (x, y) in enumerate(zip(self.RX_loc[0], self.RX_loc[1]), 0):
        #     ax.text(x, y, f'Device {i}', color='blue', fontsize=9)
        # for j, (x, y) in enumerate(zip(self.TX_loc[0], self.TX_loc[1]), 0):
        #     ax.text(x, y, f'AP {j}', color='red', fontsize=9)
        # # Draw circles around APs with radius 500
        # for x, y in zip(self.TX_loc[0], self.TX_loc[1]):
        #     circle = plt.Circle((x, y), self.R, color='red', fill=False)
        #     ax.add_artist(circle)
        # ax.set_xlim(self.RX_loc[0].min() - self.R, self.RX_loc[0].max() + self.R)
        # ax.set_ylim(self.RX_loc[1].min() - self.R, self.RX_loc[1].max() + self.R)

        #out-cell conflicts
        self.distance_vector, self.association, _ , _ = get_Device_AP_distance(self.TX_loc, self.RX_loc)
        self.g_dB2_cell2user = - (128.1 + 37.6* np.log10(0.001*self.distance_vector))
        
        scale_g_dB = - (128.1 + 37.6* np.log10(0.001 * self.R))
        self.scale_gain = np.power(10.0,scale_g_dB/10.0)

        self.db_diff_threshold = 8
        self.intercell_link_conflicts = get_conflict_links(self.association, self.g_dB2_cell2user, self.db_diff_threshold)
        #print(self.intercell_link_conflicts)
        self.intercell_link_conflicts_dict = find_all_conflict_links(self.intercell_link_conflicts, self.service_pool)
        print(self.intercell_link_conflicts_dict)
        self.neighbors = get_conflict_aps(self.intercell_link_conflicts_dict, self.cell_mapping, self.service_pool)
        print("Neighbors")
        print(self.neighbors)
        
        self.num_neighbors = list([len(nei) for nei in self.neighbors])
        self.max_num_neighbors = max(self.num_neighbors)
        print(self.max_num_neighbors)
        self.num_devices= max(list([len(self.service_pool[ap]) for ap in range(self.K)]))
        #print(self.max_deg)
        #self.obs_dim = (self.max_num_neighbors + 1) * self.num_devices * (1 + 2 * self.M)
        self.obs_dim = (self.max_num_neighbors + 1) * self.num_devices * (1 + 2 * self.M)
        self.action_dim_schedule = 1 + self.num_devices
        self.channel_random_state = np.random.RandomState(self.seed + 2024)
        self.g_dB2_cell2user = np.repeat(np.expand_dims(self.g_dB2_cell2user,axis=2),self.M,axis=-1)
        g_dB2 = np.zeros((self.N,self.N,self.M)) #gain matrix of all transmitter and receivers(links within a cell are supported by the same access point)
        for n in range(self.N):
            g_dB2[n,:,:] = self.g_dB2_cell2user[n,self.cell_mapping,:]
        self.gains = np.power(10.0, g_dB2 / 10.0)
        
    def reset(self):
        """
        # The return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        self.t = 0
        self.state_cell2user = get_random_rayleigh_variable(N = self.N, 
                                                    random_state = self.channel_random_state,
                                                    M = self.M, 
                                                    K = self.K, 
                                                    rayleigh_var = 1.0)
        tmp_H = np.zeros((self.N,self.N,self.M))
        for n in range(self.N):
            tmp_H[n,:,:] = np.multiply(np.sqrt(self.gains[n,:,:]), 
                                       abs(self.state_cell2user[n,self.cell_mapping,:]))
        self.H.append(tmp_H)
        self.p = np.zeros((self.N,self.M))
        self.SINR, self.spec_eff, self.total_interf = sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        self.create_traffic()
        self.channel_step(self.channel_random_state)
        self.load_traffic()  
        observation = self.get_state_traffic()
        sub_agent_obs = []
        for i in range(self.K):
            sub_agent_obs.append(observation[i])
        return sub_agent_obs

    def step(self, actions):
        """
        # The input of actions is a n-dimensional list, each list contains a shape = (self.action_dim, ) action data
        """
        actions = np.array(actions)
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = [False for i in range(self.K)]
        sub_agent_info = []

        self.p = self.actions2allocation(actions, self.M)
        #print(self.p)
        for n in range(self.N):
            if len(self.packets[n]) > self.unstable_th:
                sub_agent_done = [True for i in range(self.K)]
            if len(self.packets[n]) == 0:
                self.p[n,:] = 0
        self.SINR, self.spec_eff, self.total_interf = sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        for n in range(self.N):
            tmp = int(np.sum(self.spec_eff[n], axis = -1)* self.bw * self.T) 
            tmp_init = tmp
            while tmp > 0 and len(self.packets[n]) > 0:
                if tmp >= self.packets[n][0]:
                    tmp -= self.packets[n][0]
                    self.processed_packets_t[n].append(self.packets[n][0] / tmp_init + self.t - self.packets_t[n][0] - 1)
                    # print('passing')
                    del(self.packets[n][0])
                    del(self.packets_t[n][0])
                else:
                    self.packets[n][0] -= tmp
                    tmp = 0
            self.throughput[n] += tmp_init - tmp
            if not self.test:
                if len(self.packets[n]) > self.unstable_th:
                    sub_agent_done = [True for i in range(self.K)]
        self.channel_step(self.channel_random_state)
        self.load_traffic()
        rewards = self.get_reward_traffic()
        #print(self.packets)
        observations = self.get_state_traffic()
        self.t += 1
        
        if self.t >= self.max_duration:
            done = sub_agent_done = [True for i in range(self.K)]
        
        for ap in range(self.K):
            sub_agent_obs.append(observations[ap])
            sub_agent_reward.append([rewards[ap]])
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
    def create_traffic(self):
        """
        # create traffic for all the links
        """
        self.packets = [[] for i in range(self.N)]
        self.packets_t = [[] for i in range(self.N)]
        self.poisson_process = [[] for i in range(self.N)]
        if self.test:
            np.random.seed(2024)
        for i in range(self.N):
            self.poisson_process[i] = np.random.poisson(self.data_rates[i] * self.T, self.max_duration + 10)
            # self.poisson_process[i] = np.random.choice([1, 0], size=self.max_duration + 10, p=[self.data_rates[i], 1-self.data_rates[i]])
        self.processed_packets_t = [[] for i in range(self.N)]
        #print(self.poisson_process[0][:100])
    def channel_step(self, random_state = None):
        """
        # CSI update
        """
        self.state_cell2user = get_markov_rayleigh_variable(
                                state = self.state_cell2user,
                                correlation = self.correlation,
                                N = self.N,
                                random_state = random_state,
                                M = self.M, 
                                K = self.K)
        tmp_H = np.zeros((self.N,self.N,self.M))
        for n in range(self.N):
            tmp_H[n,:,:] = np.multiply(np.sqrt(self.gains[n,:,:]), abs(self.state_cell2user[n,self.cell_mapping,:]))
        self.H.append(tmp_H)
    
    def load_traffic(self):
        """
        # load traffic at each time slot(part of env transition)
        """
        for n in range(self.N):
            num_incoming = int(self.poisson_process[n][self.t])
            for i in range(num_incoming):
                self.packets[n].append(self.packet_size)
                self.packets_t[n].append(self.t)
            
    def get_state_traffic(self):
        """
        Each agent has the queue length information of itself and its neighbors,
        plus interference level and spectrum efficiency on M subbands for each device.
        Includes handling of dummy nodes with padding where devices don't exist.
        """
        local_state_dim = (self.max_num_neighbors + 1) * self.num_devices * (1 + 2 * self.M)
        local_state_entries = [np.zeros(self.num_devices * (1 + 2*self.M)) for _ in range(self.K)]
        
        for ap in range(self.K):
            offset = 0
            for local_index in range(len(self.service_pool[ap])):
                global_index = self.local_to_global.get((local_index, ap), None)
                if global_index is not None:
                    queue_length = sum(self.packets[global_index]) / self.packet_size
                    interference = np.log10(self.total_interf[global_index,:] / self.scale_gain)
                    spectrum_efficiency = self.spec_eff[global_index,:] / max_spectual_eff * 10
                    
                    # Concatenate queue length,  spectrum efficiencies and interference level
                    local_state_entries[ap][offset:offset+1+2*self.M] = np.concatenate(([queue_length], spectrum_efficiency, interference))
                else:
                    # Pad with zeros if the global index does not exist (dummy node)
                    local_state_entries[ap][offset:offset+1+2*self.M] = np.zeros(1 + self.M)
                offset += 1 + 2*self.M
                
        # np.set_printoptions(precision=4, suppress=True)
        # print(local_state_entries)
        states = [np.array([]) for _ in range(self.K)]
        for ap in range(self.K):
            states[ap] = np.array(local_state_entries[ap])
            for j in self.neighbors[ap]:
                states[ap] = np.concatenate((states[ap], local_state_entries[j]))
            if len(states[ap]) < local_state_dim:
                pad_size = local_state_dim - len(states[ap])
                states[ap] = np.pad(states[ap], (0, pad_size), 'constant', constant_values=0)
        return states

            
    def get_reward_traffic(self):     
        """
        # each agent get reward from its own utility and its neighbors(consider interaction between agents)
        """
        reward = np.zeros(self.K)
        for ap in range(self.K):
            this_reward = 0
            for device in self.service_pool[ap]:
                this_reward -=  len(self.packets[device])
            nei = self.neighbors[ap]
            for ne in nei:
                for device in self.service_pool[ne]:
                    this_reward -= len(self.packets[device]) 
            reward[ap] = this_reward    
        return reward

    def get_info_ep(self):
        """
        # compute the average delay and througput, and record the queue length at the end of each episode (get the custom metric)
        """
        queue_length = [len(self.packets[i]) for i in range(self.N)]
        #process_packets_delay = [self.processed_packets_t[i] for i in range(self.N)]
        throughput = [len(self.processed_packets_t[i]) for i in range(self.N)]
        d = []
        total_delay = 0
        num_packets = 0
        all_packets_delay = []
        for n in range(self.N):
            a = np.sum(self.processed_packets_t[n])
            all_packets_delay.extend(self.processed_packets_t[n])
            if a > 0:
                a += 1
            total_delay += a
            b = len(self.processed_packets_t[n])
            num_packets += b
            d.append(np.array(round(a / (b + non_zero_divison),2)))
    
        return {"queue_length": queue_length,
                "ave_delay":d* int(self.T*1000),
                'over_all_ave_delay' : np.array(round(total_delay / (num_packets + non_zero_divison)* int(self.T*1000),2)),
                'throughput': throughput,
                'all_packet_delay': [delay * int(self.T * 1000) for delay in all_packets_delay]}#convert from time slots to miliseconds
    
    
    def actions2allocation(self, actions, m):
        power_allocation = np.zeros((self.N, self.M))
        for ap in range(actions.shape[0]):
            subband_decisions = [actions[ap][i:i + self.action_dim_schedule + self.action_dim_power] for i in range(0, len(actions[ap]), self.action_dim_schedule + self.action_dim_power)]
            for subband_index, subband_decision in enumerate(subband_decisions):
                schedule_decision = np.argmax(subband_decision[:self.action_dim_schedule])  
                power_decision = np.argmax(subband_decision[self.action_dim_schedule:])

                if schedule_decision == 0: #indicate that no link is selected on this subbad
                    continue
                else:
                    scheduled_link_local_index = schedule_decision - 1
                    global_index = (scheduled_link_local_index, ap)
                    if global_index not in self.local_to_global:
                        continue  # consider it as dummy link
                    else:
                        # Activate the scheduled link if the key exists
                        power_allocation[self.local_to_global[global_index], subband_index] = self.powers[power_decision]
        return power_allocation
    # def actions2allocation(self, actions, m):
    #     allocation = np.zeros((self.N, self.M))
    #     for ap in range(actions.shape[0]):
    #         subband_decisions = [actions[ap][i:i+self.action_dim] for i in range(0, len(actions[ap]), self.action_dim)]
    #         for subband_index, subband_decision in enumerate(subband_decisions):
    #             action = np.argmax(subband_decision)
    #             if action == 0: #indicate that no link is selected on this subbad
    #                 continue
    #             else:
    #                 scheduled_link_local_index = action - 1
                    
    #                 global_index = (scheduled_link_local_index, ap)
    #                 if global_index not in self.local_to_global:
    #                     continue  # consider it as dummy link
    #                 else:
    #                     # Activate the scheduled link if the key exists
    #                     allocation[self.local_to_global[global_index], subband_index] = 1
    #     return allocation
    