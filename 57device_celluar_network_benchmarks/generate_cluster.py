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
    # Store the required information in a dictionary for future reference
    return {
        "overall_average_delay": info['over_all_ave_delay'],
        "queue_length": info['queue_length']
    }

M = 3
N = 57
random_deployment = True
drs = [20, 25, 30]  # Different values of dr
# random_deployment = False
# drs = [15, 20, 25]  # Different values of dr
methods = ['FP', 'WMMSE', 'FITLQ', 'shared_nn','separate_nn', 'LLQ']

# Load and store data for each dr and method
overall_average_delays = {dr: {} for dr in drs}

# Iterate over each dr to load data
for dr in drs:
    data_file_path = f'ch{M}_{N}links_dr{dr}_rd{random_deployment}'
    method_paths = {
        'separate_nn': 'separate_' + data_file_path + '.npz',
        'shared_nn': 'shared_' + data_file_path + '.npz',
        'FP' : 'FP_' + data_file_path + '.npz',
        'FITLQ': 'FITLQ_' + data_file_path + '.npz',
        'WMMSE': 'wmmse_' + data_file_path + '.npz',
        'LLQ': 'LLQ2_' + data_file_path + '.npz'
    }
    
    for method, path in method_paths.items():
        method_data = load_data(path)
        info = retrieve_info(method_data[0])
        overall_average_delays[dr][method] = info['overall_average_delay']

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.15
opacity = 0.8
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
labels = ['FP', 'WMMSE', 'ITLQ', 'shared_nn', 'separate_nn', 'LLQ']
hatches = ['/', '\\', '|', '-', '+', 'x']
y_lim = 70
# Set x-axis labels
index = np.arange(len(drs))

# Plot bars for each method at each dr value
for i, method in enumerate(methods):
    delays = [overall_average_delays[dr][method] for dr in drs]
    rects = ax.bar(index + i * bar_width, delays, bar_width, 
                   alpha=opacity, color=colors[i], label=labels[i],
                   edgecolor='black', hatch=hatches[i])
    # Add text labels above the bars
    for rect, delay in zip(rects, delays):
        y_position = delay if delay <= y_lim else y_lim  # Adjust y position if delay exceeds y-limit
        text_offset = (3 if delay <= y_lim else -20)  # Place text above or below the top of the bar based on height
        ax.annotate(f'{delay:.2f}', xy=(rect.get_x() + rect.get_width() / 2, y_position),
                    xytext=(0, text_offset),  # Dynamic offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=12)


# Set labels, titles, and legends
ax.set_xlabel('Traffic Load', fontsize=18)
ax.set_ylabel('Overall Average Delay', fontsize=18)
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(('light traffic', 'medium traffic', 'heavy traffic'), fontsize=16)
ax.set_ylim(0, y_lim)  # Apply y-axis limit
ax.legend()
ax.legend(fontsize=16)
plt.tight_layout()
plt.savefig('./figures/{}cluster_delay.eps'.format(data_file_path),format='eps',dpi=1600)
plt.show()











