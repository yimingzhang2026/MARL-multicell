import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random
import copy
import math

non_zero_divison = 1e-7

class env(object):
    """
    # env agent def
    """

    def __init__(self,args):
        self.N = args.N
        self.K = args.K
        self.M = args.M
        self.max_duration = args.episode_length
        self.data_rates = args.data_rates
        self.seed = args.seed
        self.unstable_th = args.max_queue
        np.random.seed(self.seed)
        if self.N == 8 and self.K == 4:
            G = nx.Graph()
            # Add nodes for 8 devices and 4 APs
            self.nodes = list(range(self.N))
            # Add nodes to the graph
            G.add_nodes_from(self.nodes)
            # Set positions for the APs and devices
            pos = {
                0: (4, 2), 1: (2, 2),  # Devices under AP1
                2: (1, 1), 3: (1, -1),  # Devices under AP2
                4: (2, -2), 5: (4, -2),  # Devices under AP3
                6: (5, -1), 7: (5, 1),  # Devices under AP4
            }
            DG = nx.DiGraph(G)
    
            # Adjust edges to use integer node identifiers
            bidirectional_edges = [(0,1), (2,3),(4,5), (6,7), (1, 2), (3, 4), (5, 6), (7, 0)]
            DG.add_edges_from(bidirectional_edges)
            DG.add_edges_from([(b, a) for a, b in bidirectional_edges])
            directional_edges = [(1, 7), (7, 5), (5, 3), (3, 1), (0, 2), (2, 4), (4, 6), (6, 0)]
            DG.add_edges_from(directional_edges)
    
            self.conflicts = self.get_interference_dict(DG)
            print(self.conflicts)
            plt.figure(figsize=(8, 8))
            nx.draw(DG, pos=pos, with_labels=True, node_size=700, node_color="skyblue", font_size=15, font_weight="bold", arrows=True)
            plt.margins(0.1)
            plt.gca().axis('off')
        else:
            print("Deployment not implemented yet")

        self.neighbors=[[] for _ in range(self.K)]
        self.neighbors[0] = [1,3]
        self.neighbors[1] = [0,2]
        self.neighbors[2] = [1,3]
        self.neighbors[3] = [2,0]
        self.cell_mapping = [0, 0, 1, 1, 2, 2, 3, 3]
        self.service_pool = [[0, 1], [2, 3], [4, 5], [6, 7]]
        self.num_devices= 2
        self.global_to_local, self.local_to_global = self.global_local_index_mapping(self.service_pool)
        #example of bijection mappings
        #self.local_to_global[(1, 2)], self.global_to_local[5]

    def reset(self):
        """
        # The return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        self.t = 0
        self.create_traffic()
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
        
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = [False for i in range(self.K)]
        sub_agent_info = []
        allocated_spectrum = np.zeros((self.N,self.M))
        for n in range(self.N):
            if len(self.packets[n]) == 0:
                allocated_spectrum[n,:] = 0
        for ap in range(self.K):
            for local_index in range(actions[ap].shape[0]):
                global_index = (local_index, ap)
                if global_index not in self.local_to_global:
                    continue  # consider it as dummy link
                else:
                    allocated_spectrum[self.local_to_global[global_index],:] = actions[ap][local_index,:]
        # print("allocated")
        # print(allocated_spectrum)
        physical_spectrum = self.resolve_conflicts(allocated_spectrum, self.conflicts)
        # print("physical")
        # print(physical_spectrum)

        for n in range(self.N):
            tmp = np.sum(physical_spectrum[n,:])
            tmp_init = tmp
            while tmp > 0 and len(self.packets[n]) > 0:
                if tmp >= self.packets[n][0]:
                    tmp -= self.packets[n][0]
                    self.processed_packets_t[n].append(self.t - self.packets_t[n][0])
                    del(self.packets[n][0])
                    del(self.packets_t[n][0])
                else:
                    self.packets[n][0] -= tmp
                    tmp = 0
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
        # create traffic for all the agents
        """
        # if self.seed:
        #     np.random.seed(self.seed)
        self.packets = [[] for i in range(self.N)]
        self.packets_t = [[] for i in range(self.N)]
        
        self.poisson_process = [[] for i in range(self.N)]
        for i in range(self.N):
            self.poisson_process[i] = np.random.poisson(self.data_rates[i], self.max_duration + 10)
            
        self.processed_packets_t = [[] for i in range(self.N)]
    
    def load_traffic(self):
        """
        # load traffic at each time slot(part of env transition)
        """
        for n in range(self.N):
            num_incoming = int(self.poisson_process[n][self.t])
            while num_incoming != 0:
                self.packets[n].append(1)
                self.packets_t[n].append(self.t)
                num_incoming -= 1
            
            
    def get_state_traffic(self):
        """
        # each agent has the queue length information of itself and its neighbors
        """
        # Initialize the states as a list of numpy arrays instead of a list of lists
        states = [np.array([]) for _ in range(self.K)]
        local_state_entries = [np.zeros(len(self.service_pool[ap])) for ap in range(self.K)]
        for ap in range(self.K):
            for local_index in range(len(local_state_entries[ap])):
                local_state_entries[ap][local_index] = len(self.packets[self.local_to_global[(local_index, ap)]])
    
        for ap in range(self.K):
            states[ap] = np.array(local_state_entries[ap])
            nei = copy.copy(self.neighbors[ap])
            for j in nei:
                states[ap] = np.concatenate((states[ap], local_state_entries[j]))

        return states
    
    def get_reward_traffic(self):     
        """
        # each agent get reward from its own utility and its neighbors(consider interaction between agents)
        """
        reward = np.zeros(self.K)
        for ap in range(self.K):
            this_reward = 0
            for device in self.service_pool[ap]:
                this_reward -= len(self.packets[device])
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
                "ave_delay":d,
                'over_all_ave_delay' : np.array(round(total_delay / (num_packets + non_zero_divison),2)),
                'throughput': throughput,
                'all_packet_delay': all_packets_delay}
    
    def get_interference_dict(self,directed_graph):
        """
        Generates a dictionary indicating which nodes are interfered by other nodes.
        
        Parameters:
        - directed_graph: A NetworkX DiGraph object representing the network.
        
        Returns:
        - A dictionary where keys are node labels and values are lists of nodes that interfere with the key node.
        """
        interference_dict = {}
        for node in directed_graph.nodes():
            interfered_by = [source for source, target in directed_graph.in_edges(node)]
            if interfered_by:  # Only add to the dictionary if there are interfering devices
                interference_dict[node] = interfered_by
        return interference_dict
    def global_local_index_mapping(self,service_pool):
        # Initialize mappings for both directions
        global_to_local= {}  # From the previous step
        local_to_global = {}  # New mapping
        
        # Populate the mappings
        for pool_id, devices in enumerate(service_pool):
            for local_index, global_index in enumerate(devices):
                # Update mapping from global to local and service pool
                global_to_local[global_index] = (local_index, pool_id)
                
                # Update mapping from local and service pool to global
                local_to_global[(local_index, pool_id)] = global_index
        
        return global_to_local, local_to_global
    def resolve_conflicts(self, matrix, conflicts):
        # Create a copy of the matrix to modify
        resolved_matrix = copy.copy(matrix)
        # Iterate through each link and its conflicts
        for link, conflicting_links in conflicts.items():
            for subband in range(matrix.shape[1]):
                if matrix[link][subband] == 1:
                    # Check each conflicting link
                    for conflict_link in conflicting_links:
                        if matrix[conflict_link][subband] == 1:
                            resolved_matrix[link][subband] = 0
        return resolved_matrix
        