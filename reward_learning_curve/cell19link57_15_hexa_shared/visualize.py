# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 23:38:19 2023

@author: zyimi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Function to calculate exponential moving average (EMA)
def calculate_ema(values, alpha=0.1):
    ema_values = []
    ema = values[0]  # Starting the EMA with the first data point

    for value in values:
        ema = alpha * value + (1 - alpha) * ema
        ema_values.append(ema)

    return ema_values

# Load the CSV file
file_path = './average_episode_rewards_average_episode_rewards.csv'
data_df = pd.read_csv(file_path)

# Extracting the 'Step' and 'Value' columns for plotting
steps_provided = data_df['Step']
values_provided = data_df['Value']

# Calculate EMA for the provided data
ema_values = calculate_ema(values_provided)

line_width = 2.5
# Plotting the curves
fig, ax = plt.subplots(figsize=(10, 6))

# Original curve
plt.plot(steps_provided, values_provided, label='Original', linestyle='--', linewidth = line_width)

# EMA smoothed curve
plt.plot(steps_provided, ema_values, label='Smoothed (EMA)', color='orange', linewidth = line_width)

# Adding labels and title
plt.xlabel('Time slots (millions)', fontsize = 20)
plt.ylabel('Average episode reward', fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(fontsize = 20)
plt.xlim(-1e5,2e6)

def millions_formatter(x, pos):
    return f'{x/1e6:.1f}'
ax.xaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))

plt.grid(True)
plt.tight_layout()
plt.savefig('./eval_reward_3_57links_shared.png', format='png',dpi = 1600)
plt.savefig('./eval_reward_3_57links_shared.eps', format='eps',dpi = 600)
# Show the plot
plt.show()