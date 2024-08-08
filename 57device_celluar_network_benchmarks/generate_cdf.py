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

def cdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def plot_cdf(a, label=None, color=None, linestyle=None, marker=None):
    x, y = cdf(a)
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    plt.plot(x, y, drawstyle='steps-post', label=label, color=color, linestyle=linestyle, marker=marker, markevery=0.1)
    plt.grid(True)

def retrieve_info(info):
    print(f"Queue Length: {info['queue_length']}")
    print(f"Overall Average Delay: {info['over_all_ave_delay']}")
    stored_info = {
        "over_all_ave_delay": info['over_all_ave_delay'],
        "average_delay_link": info["ave_delay"],
        "throughput": info["throughput"],
        "all_packet_delay": info["all_packet_delay"]
    }
    return stored_info

def plot_multiple_datasets(file_paths, labels, colors, linestyles, markers):
    plt.figure(figsize=(10, 6))
    overall_avg_delays = []
    for file_path, label, color, linestyle, marker in zip(file_paths, labels, colors, linestyles, markers):
        data = load_data(file_path)
        info = retrieve_info(data[0])
        plot_cdf(info["all_packet_delay"], label=label, color=color, linestyle=linestyle, marker=marker)
        overall_avg_delays.append(info["over_all_ave_delay"])

    
        # Print max delay information
        max_delay = max(info["all_packet_delay"])
        print(f"{label} Max Delay: {max_delay:.2f}")
    avg_delay_str = 'Average packet delay\n' + '\n'.join([f'{label}: {oad:.2f}' for label, oad in zip(labels, overall_avg_delays)])
    plt.figtext(0.65, 0.45, avg_delay_str, ha="left", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.legend(loc='lower right', fontsize=12)
    plt.xlabel('Packet Delay (ms)', fontsize=16)
    plt.ylabel('CDF', fontsize=16)
    plt.xlim(-1, 150)
    plt.savefig('./figures/{}cdf_delay.eps'.format(data_file_path),format='eps',dpi=1600)
    plt.show()

# Example usage
M = 3
N = 57
dr = 25
randomDep = 'True'
data_file_path = f'ch{M}_{N}links_dr{dr}_rd{randomDep}'
# Define paths
shared_path = 'shared_' + data_file_path + '.npz'
separate_path = 'separate_' + data_file_path + '.npz'
Greedy_path = 'LLQ2_' + data_file_path + '.npz'
fp_path = 'FP_' + data_file_path + '.npz'
wmmse_path = 'WMMSE_' + data_file_path + '.npz'
FITLQ_path = "FITLQ_" + data_file_path + '.npz'
file_paths = [fp_path, wmmse_path, FITLQ_path, Greedy_path, shared_path, separate_path]
labels = ['FP', 'WMMSE', 'ITLinQ',  'Greedy', 'Shared NN', 'separate NN'] 
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']#'##e377c2', '#7f7f7f', '#bcbd22', '#17becf'
linestyles = [':', '--', '-.', '-.', '-', '-']
markers = ['>', '<', '^', 'D', 's', 'p']
plot_multiple_datasets(file_paths, labels, colors, linestyles, markers)

















