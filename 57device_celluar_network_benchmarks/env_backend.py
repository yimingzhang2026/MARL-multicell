# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:20:38 2023

@author: zyimi
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from itertools import cycle
from itertools import combinations
from collections import OrderedDict
import networkx as nx
import copy

max_SINR = 10*np.log10(1000)
min_SINR = -20
max_spectual_eff = np.log2(1.0+1000)

def voronoi_finite_polygons_2d(vor, radius=None):

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def generate_deployment_random(R, K, N, seed):
    max_dist = R
    TX_loc = np.zeros((2,K))
    RX_loc = np.zeros((2,N))
    APs_pool = cycle([i for i in range(K) for _ in range(N//K)])
    generated_APs = 0
    i = 0
    #deploy APs
    TX_loc [0, generated_APs] = 0.0
    TX_loc [1, generated_APs] = 0.0
    generated_APs += 1
    # generate fix AP location using random seed
    random_state = np.random.RandomState(seed)

    while(generated_APs < K):
        for j in range(6):
            tmp_xloc = TX_loc [0, i]+np.sqrt(3)*R*np.cos(j*np.pi/(3)) +  0.5 * R * random_state.rand()
            tmp_yloc = TX_loc [1, i]+np.sqrt(3)*R*np.sin(j*np.pi/(3)) +  0.5 * R * random_state.rand() 
            was_before = False
            for inner_loop in range(generated_APs):
                if (abs(tmp_xloc-TX_loc [0, inner_loop*1])<R*1e-2 and abs(tmp_yloc-TX_loc [1, inner_loop*1])<R*1e-2):
                    was_before = True
                    break
            if (not was_before):
                TX_loc [0, generated_APs] = tmp_xloc
                TX_loc [1, generated_APs] = tmp_yloc
                generated_APs += 1
            if(generated_APs>= K):
                break
        i += 1
    low_bound_x = min(TX_loc[0,:]) - 0.5 * R
    upper_bound_x = max(TX_loc[0,:]) + 0.5 * R
    low_bound_y = min(TX_loc[1,:]) - 0.5 * R
    upper_bound_y = max(TX_loc[1,:]) + 0.5 * R
    
    np.random.seed(seed)
    for i in range(N):   
        ap = next(APs_pool)
        center_x = TX_loc[0,ap]
        center_y = TX_loc[1,ap]
        RX_loc[0,i]= np.random.uniform(center_x - 0.5 * R,center_x + 0.5 * R)
        RX_loc[1,i]= np.random.uniform(center_y - 0.5 * R,center_y + 0.5 * R)
        
#%%plot part, comment out in training to speed up
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1)   
    if K > 2 :
    #Generate Voronoi graph
        vor = Voronoi(np.transpose(TX_loc))
        regions, vertices = voronoi_finite_polygons_2d(vor, radius = 5*R)
        polygons = []
        for reg in regions:
            polygon = vertices[reg]
            polygons.append(polygon)
        
        for poly in polygons:
            p = Polygon(poly, edgecolor = (0.5, 0.1, 0.1), fill = False)   
            ax.add_patch(p)
            ax.set_xlim(low_bound_x , upper_bound_x)
            ax.set_ylim(low_bound_y , upper_bound_y)
        

    for i in range(1):
        plt.plot(TX_loc[0,i],TX_loc[1,i],'g^', label = 'Base Station')
        plt.text(TX_loc[0,i],TX_loc[1,i], 'AP{}'.format(i+1), fontsize=12)
        #circ = plt.Circle((TX_loc[0,i],TX_loc[1,i]),min_dist,color='k',fill=False)
        #ax.add_patch(circ)
    for i in range(1,K):
        plt.plot(TX_loc[0,i],TX_loc[1,i],'g^')
        plt.text(TX_loc[0,i],TX_loc[1,i], 'AP{}'.format(i+1), fontsize=12)
        #circ = plt.Circle((TX_loc[0,i],TX_loc[1,i]),min_dist,color='k',fill=False)
        #ax.add_patch(circ)

    for i in range(1):
        plt.plot(RX_loc[0,i],RX_loc[1,i],'ro',label = 'Device')
        plt.text(RX_loc[0,i],RX_loc[1,i], 'D{}'.format(i+1), fontsize=12)
    for i in range(1,N): 
        plt.plot(RX_loc[0,i],RX_loc[1,i],'ro')     
        plt.text(RX_loc[0,i],RX_loc[1,i], 'D{}'.format(i+1), fontsize=12)
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))   
    ax.legend(fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.xlabel('x axis position (meters)',fontsize=16)
    plt.ylabel('y axis position (meters)',fontsize=16)
    # plt.savefig('./deployment_K{}N{}seed{}_random.png'.format(K,N,seed), format='png', dpi=1500)
    plt.savefig('./deployment_K{}N{}seed{}_random.eps'.format(K,N,seed), format='eps', dpi=1600)
    plt.show()    
#%%
    _, _, cell_mapping, service_pool = get_Device_AP_distance(TX_loc, RX_loc)
    return TX_loc, RX_loc, cell_mapping, service_pool


def generate_deployment_hexa(R, K, N, seed):
    assert N % K == 0, 'N needs to be divisible by K!'
    
    max_dist = R
    min_dist = 35
    x_hexagon=R*np.array([0, -np.sqrt(3)/2, -np.sqrt(3)/2, 0, np.sqrt(3)/2, np.sqrt(3)/2, 0])
    y_hexagon=R*np.array([-1, -0.5, 0.5, 1, 0.5, -0.5, -1])
    
    TX_loc = np.zeros((2,K))
    RX_loc = np.zeros((2,N))
    TX_xhex = np.zeros((7,K))
    TX_yhex = np.zeros((7,K))
    
    cell_mapping = np.zeros(N).astype(int)
    service_pool = [[] for _ in range(K)]

    APs_pool = cycle([i for i in range(K) for _ in range(N//K)])
    generated_APs = 0
    i = 0
    #deploy APs
    TX_loc [0, generated_APs] = 0.0
    TX_loc [1, generated_APs] = 0.0
    TX_xhex [:,generated_APs] = x_hexagon
    TX_yhex [:,generated_APs] = y_hexagon
    generated_APs += 1
    # generate fix AP location using random seed
    random_state = np.random.RandomState(seed)

    while(generated_APs < K):
        for j in range(6):
            tmp_xloc = TX_loc [0, i]+np.sqrt(3)*R*np.cos(j*np.pi/(3)) 
            tmp_yloc = TX_loc [1, i]+np.sqrt(3)*R*np.sin(j*np.pi/(3)) 
            tmp_xhex = tmp_xloc+x_hexagon
            tmp_yhex = tmp_yloc+y_hexagon
            was_before = False
            for inner_loop in range(generated_APs):
                if (abs(tmp_xloc-TX_loc [0, inner_loop*1])<R*1e-2 and abs(tmp_yloc-TX_loc [1, inner_loop*1])<R*1e-2):
                    was_before = True
                    break
            if (not was_before):
                TX_loc [0, generated_APs] = tmp_xloc
                TX_loc [1, generated_APs] = tmp_yloc
                TX_xhex [:,generated_APs] = tmp_xhex
                TX_yhex [:,generated_APs] = tmp_yhex   
                generated_APs += 1
            if(generated_APs>= K):
                break
        i += 1
    for i in range(N):
        ap = next(APs_pool)
        cell_mapping[i] = ap
        service_pool[ap].append(i)
        this_cell = cell_mapping[i]
        # Place UE within that cell.
        constraint_minx_UE=min(TX_xhex[:,this_cell])
        constraint_maxx_UE=max(TX_xhex[:,this_cell])
        constraint_miny_UE=min(TX_yhex[:,this_cell])
        constraint_maxy_UE=max(TX_yhex[:,this_cell])
        inside_checker = True
    
        while (inside_checker):
            RX_loc[0,i]= np.random.uniform(constraint_minx_UE,constraint_maxx_UE)
            RX_loc[1,i]= np.random.uniform(constraint_miny_UE,constraint_maxy_UE)
            tmp_distance2center = np.sqrt(np.square(RX_loc[0,i]-TX_loc [0, this_cell])+np.square(RX_loc[1,i]-TX_loc [1, this_cell]))
            if(_inside_hexagon(RX_loc[0,i],RX_loc[1,i],TX_xhex[:,this_cell],TX_yhex[:,this_cell])
                and tmp_distance2center>min_dist and tmp_distance2center<max_dist):
                inside_checker = False

#%%plot part, comment out in training to speed up
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1)   

    # Plotting APs
    for i in range(K):
        plt.plot(TX_loc[0,i],TX_loc[1,i],'g^')
        plt.text(TX_loc[0,i],TX_loc[1,i], 'AP{}'.format(i+1), fontsize=12)
        # Plot the hexagonal boundaries for each AP
        plt.plot(np.append(TX_xhex[:,i], TX_xhex[0,i]), np.append(TX_yhex[:,i], TX_yhex[0,i]), 'g-')
    
    # Plotting Devices
    for i in range(N):
        plt.plot(RX_loc[0,i],RX_loc[1,i],'ro')
        plt.text(RX_loc[0,i],RX_loc[1,i], 'D{}'.format(i+1), fontsize=12)
    
    # Setting legend for the first AP and Device to avoid repetition in the legend
    plt.plot(TX_loc[0,0],TX_loc[1,0],'g^', label='Base Station')
    plt.plot(RX_loc[0,0],RX_loc[1,0],'ro', label='Device')
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))   
    ax.legend(fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.xlabel('x axis position (meters)',fontsize=16)
    plt.ylabel('y axis position (meters)',fontsize=16)
    # plt.savefig('./deployment_K{}N{}seed{}_hexa.png'.format(K,N,seed), format='png', dpi=1500)
    plt.savefig('./deployment_K{}N{}seed{}_hexa.eps'.format(K,N,seed), format='eps', dpi=1600)
    plt.show()    
#%%
    return TX_loc, RX_loc, cell_mapping, service_pool


def get_Device_AP_distance(TX_loc, RX_loc):
    N = len(RX_loc[0])
    K = len(TX_loc[0])
    
    cell_mapping = np.zeros(N).astype(int)
    service_pool = [[] for idx in range(K)]
    
    distance_vector = np.zeros((N,K))
    association_vector = np.zeros((N,K))
    
    for ap in range(K):
        for device in range (N):
            distance_vector[device,ap] = np.sqrt(np.square(RX_loc[0,device]-TX_loc [0, ap])+np.square(RX_loc[1,device]-TX_loc [1, ap]))
    
    for device in range(N):
        ap = np.argmin(distance_vector[device,:])
        association_vector[device][ap] = 1
        #print("device{} is served by AP{}".format(device,ap))
        cell_mapping[device] = ap
        service_pool[ap].append(device)
    return distance_vector, association_vector, cell_mapping, service_pool

def _inside_hexagon(x,y,TX_xhex,TX_yhex):
    n = len(TX_xhex)-1
    inside = False
    p1x,p1y = TX_xhex[0],TX_yhex[0]
    for i in range(n+1):
        p2x,p2y = TX_xhex[i % n],TX_yhex[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def get_conflict_links(association, gain, db_diff):
    N = len(gain)
    K = len(gain[0])
    association = np.array(association)
    gain = np.array(gain)
    conflicts = []
    for n in range(N):
        ap = np.argmax(association[n,:])
        for k in range(K):
            if k != ap and gain[n,ap] - gain[n,k] < db_diff and sum(association[:,k]) > 0:
                #print(k) # possible APs that may cause interference
                c = np.argwhere(association[:,k] == 1).flatten().tolist()
                c.append(n)
                edges = list(combinations(c, 2))
                for x in edges:
                    conflicts.append(x)
    conflicts = set(conflicts)  # Use a set to eliminate duplicates
    for (a, b) in list(conflicts):
        if np.argmax(association[a, :]) == np.argmax(association[b, :]):
            # Both nodes are served by the same AP, remove this conflict
            conflicts.discard((a, b))

    return list(conflicts)
# def in_cell_conflicts(service_pool):
#     in_cell_conflicts = []
#     for links in service_pool:
#         if len(links) > 1:#more than 1 link in the pool
#             edges = list(combinations(links, 2))
#             for x in edges:
#                 in_cell_conflicts.append(x)
#     return in_cell_conflicts
    
def find_all_conflict_links(conflicts, service_pool):
    # Dictionary to hold all links and their corresponding conflicting links
    conflict_dict = {}

    # Check each pair to see if the link is part of the pair and add the other element of the pair to the set
    for a, b in conflicts:
        if a not in conflict_dict:
            conflict_dict[a] = set()
        if b not in conflict_dict:
            conflict_dict[b] = set()
        
        conflict_dict[a].add(b)
        conflict_dict[b].add(a)
    
    # Convert sets to sorted lists
    sorted_keys = sorted(conflict_dict.keys())
    sorted_conflict_dict = {key: sorted(conflict_dict[key]) for key in sorted_keys}

    return sorted_conflict_dict


def get_conflict_aps(intercell_conflict_links, cell_mapping, service_pool):
    K = len(service_pool)
    conflict_ap_neighbors=[[] for _ in range(K)]
    for ap in range(K):#find all the aps it has conflict with
        conflict_links = []
        conflict_aps = []
        for ue in service_pool[ap]:
            if ue in intercell_conflict_links:
                for link in intercell_conflict_links[ue]:
                    if link not in conflict_links:
                        conflict_links.append(link)
        for conflict_link in conflict_links:
            if cell_mapping[conflict_link] not in conflict_aps and cell_mapping[conflict_link] != ap:
                conflict_aps.append(cell_mapping[conflict_link])
        conflict_ap_neighbors[ap] = conflict_aps
    return conflict_ap_neighbors


def global_local_index_mapping(service_pool):
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


def get_random_rayleigh_variable(N,
                                  K,
                                  random_state = None,
                                  M=1, 
                                  rayleigh_var=1.0):
    if random_state is None:
        return np.sqrt(2.0/np.pi) * (rayleigh_var * np.random.randn(N, K, M) +
                                                1j * rayleigh_var * random_state.randn(N, N, M))
    else:
        return np.sqrt(2.0/np.pi) * (rayleigh_var * random_state.randn(N, K, M) +
                                                1j * rayleigh_var * random_state.randn(N, K, M))
    
def get_markov_rayleigh_variable(state,
                                  correlation,
                                  N,
                                  K,
                                  random_state = None,
                                  M=1, 
                                  rayleigh_var=1.0):
    if random_state is None:
        return correlation*state +np.sqrt(1-np.square(correlation)) * np.sqrt(2.0/np.pi) * (rayleigh_var * np.random.randn(N, K, M) +
                                                1j * rayleigh_var * np.random.randn(N, K, M))
    else:
        return correlation*state +np.sqrt(1-np.square(correlation)) * np.sqrt(2.0/np.pi) * (rayleigh_var * random_state.randn(N, K, M) +
                                                1j * rayleigh_var * random_state.randn(N, K, M))
    
def sumrate_multi_list_clipped(H,p,noise_var):
    # Take pow 2 of abs_H, no need to take it again and again
    H_2 = H ** 2
    N = H.shape[1] # number of links
    M = H.shape[2] # number of channels
    
    sum_rate1 = np.zeros((N, M)) #calculate SINR
    sum_rate2 = np.zeros((N, M)) #calculate spec_eff
    total_interf = np.zeros((N, M))
    for out_loop in range(M):
        for loop in range (N):
            tmp_1 = H_2[loop, loop, out_loop] * p[loop, out_loop]
            tmp_2 = np.matmul(H_2[loop, :, out_loop], p[:, out_loop]) + noise_var - tmp_1
            total_interf[loop,out_loop] = tmp_2
            if tmp_1 == 0:
                sum_rate1[loop,out_loop] = 0.0
                sum_rate2[loop,out_loop] = 0.0
            else:
                sum_rate1[loop,out_loop] = 10*np.log10(tmp_1/tmp_2)
                sum_rate2[loop,out_loop] = np.log2(1.0+tmp_1/tmp_2)
    sum_rate1 = np.clip(sum_rate1, a_min = min_SINR, a_max = max_SINR)
    sum_rate2 = np.clip(sum_rate2, a_min = None, a_max = max_spectual_eff)
    return sum_rate1, sum_rate2, total_interf

def calculate_snr_inr(H, p, noise_var):
    """
    Calculate the SNR and INR matrices.

    Parameters:
    H (numpy.ndarray): Channel gain matrix of shape (N, N, M)
    p (numpy.ndarray): Power allocation matrix of shape (N, M)
    noise_var (float): Noise variance

    Returns:
    snr (numpy.ndarray): SNR matrix of shape (N, M)
    inr (numpy.ndarray): INR matrix of shape (N, N, M)
    total_interf (numpy.ndarray): Total interference matrix of shape (N, M)
    """
    H_2 = H ** 2  # Squared channel gains
    N, _, M = H.shape  # Number of links and channels
    
    snr = np.zeros((N, M))  # Initialize SNR matrix
    inr = np.zeros((N, N, M))  # Initialize INR matrix
    total_interf = np.zeros((N, M))  # Initialize total interference matrix

    for out_loop in range(M):
        for loop in range(N):
            desired_signal_power = H_2[loop, loop, out_loop] * p[loop, out_loop]
            total_interference_power = np.sum(H_2[:, loop, out_loop] * p[:, out_loop]) - desired_signal_power
            total_noise_interference_power = total_interference_power + noise_var
            
            total_interf[loop, out_loop] = total_interference_power
            
            if desired_signal_power == 0:
                snr[loop, out_loop] = 0.0
            else:
                snr[loop, out_loop] = desired_signal_power / noise_var

            for interferer in range(N):
                if interferer != loop:
                    inr[loop, interferer, out_loop] = (H_2[interferer, loop, out_loop] * p[interferer, out_loop]) / noise_var
    
    return snr, inr