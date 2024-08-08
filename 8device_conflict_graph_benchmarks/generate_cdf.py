# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:23:49 2023

@author: zyimi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def load_data(file_path):
    loaded_data = np.load(file_path, allow_pickle=True)
    data = [item.item() for item in loaded_data.values()]
    return data

M = 3
N = 8
dr = 0.3
# policy = 'shared'
#policy = 'separate'
data_file_path = f'ch{M}_{N}nodes_dr{dr}'
shared_nn_path = 'shared_' +'nn_' + data_file_path + '.npz'
separate_nn_path = 'separate_' +'nn_' + data_file_path + '.npz'
benchmark_path = 'benchmark_' + data_file_path + '.npz'
QCSMA_path = 'qcsma_' + data_file_path + '.npz'
GMS_path = 'gms_' + data_file_path + '.npz'
average_delay = {}
def cdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def plot_cdf(a):
    x, y = cdf(a)
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    plt.plot(x, y, drawstyle='steps-post')
    plt.grid(True)

def retrieve_info(info):
 
    # Print the required information
    print(f"Queue Length: {info['queue_length']}")
    print(f"Overall Average Delay: {info['over_all_ave_delay']}")

    # Store the remaining information in a dictionary for future reference
    stored_info = {
        "over_all_ave_delay": info['over_all_ave_delay'],
        "average_delay_link": info["ave_delay"],
        "throughput": info["throughput"],
        "all_packet_delay": info["all_packet_delay"]
    }

    # The stored_info can be returned or used as needed
    return stored_info


benchmark_data = load_data(benchmark_path)
benchmark_info = retrieve_info(benchmark_data[0]) #retrive the information of 1st experiment
plot_cdf(benchmark_info["all_packet_delay"])

QCSMA_data = load_data(QCSMA_path)
QCSMA_info = retrieve_info(QCSMA_data[0]) #retrive the information of 1st experiment
plot_cdf(QCSMA_info["all_packet_delay"])

GMS_data = load_data(GMS_path)
GMS_info = retrieve_info(GMS_data[0]) #retrive the information of 1st experiment
plot_cdf(GMS_info["all_packet_delay"])


shared_nn_data = load_data(shared_nn_path)
shared_nn_info = retrieve_info(shared_nn_data[0]) #retrive the information of 1st experiment
plot_cdf(shared_nn_info["all_packet_delay"])

separate_nn_data = load_data(separate_nn_path)
separate_nn_info = retrieve_info(separate_nn_data[0]) #retrive the information of 1st experiment
plot_cdf(separate_nn_info["all_packet_delay"])

plt.figure(figsize=(10, 6))
colors = ['navy','lightblue', 'orange', 'green', 'purple']
for info, label, color in zip([GMS_info, benchmark_info, QCSMA_info, shared_nn_info, separate_nn_info ], ['GMS','LLQ', 'QCSMA', 'shared_nn', 'separate_nn'], colors):
    x, y = cdf(info["all_packet_delay"])
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    plt.plot(x, y, drawstyle='steps-post', label=label,color=color)
    # Highlighting the maximum packet delay
    plt.axvline(x=max(x), linestyle='--',color=color, label=f'{label} Max Delay: {max(x):.2f}')
    plt.legend(fontsize=18)

# Adding legend and axis labels
plt.legend()
plt.grid(True)
plt.xlabel('Packet Delay (in time slots)', fontsize=16)
plt.ylabel('CDF', fontsize=16)

# Show overall average delays in the plot
overall_avg_delays = [benchmark_info["over_all_ave_delay"], QCSMA_info["over_all_ave_delay"], shared_nn_info["over_all_ave_delay"], separate_nn_info["over_all_ave_delay"], GMS_info["over_all_ave_delay"]]
avg_delay_str = 'Average packet delay\n' + '\n'.join([f'{label}: {oad:.2f}' for label, oad in zip(['LLQ', 'QCSMA', 'shared_nn', 'separate_nn', 'GMS'], overall_avg_delays)])
plt.figtext(0.65, 0.55, avg_delay_str, ha="left", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.savefig('./figures/{}cdf.eps'.format(data_file_path), format='eps',dpi=1600)
plt.show()














