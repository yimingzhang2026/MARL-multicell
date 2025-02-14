import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_agent_rewards():
    # Initialize an empty list to store DataFrames
    dfs = []
    # Load CSV files and drop the 'Wall time' column
    for i in range(19):
        try:
            file_path = f"./agent{i}_eval_average_episode_rewards_agent{i}_eval_average_episode_rewards.csv"
            df = pd.read_csv(file_path)
            df = df.drop('Wall time', axis=1)
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: Could not find file {file_path}")
            continue

    if not dfs:
        print("No data files were loaded successfully.")
        return

    # Create a new DataFrame for plotting, starting with the 'Step' column
    new_df = pd.DataFrame(dfs[0]['Step'], columns=['Step'])
    
    # Extract and rename the 'Value' column for each agent
    for i, df in enumerate(dfs):
        agent_col_name = f'Agent {i+1}'
        new_df[agent_col_name] = df['Value']

    # Calculate the average reward across all agents for each step
    new_df['Average Reward'] = new_df.iloc[:, 1:].mean(axis=1)

    # Create the plot with larger figure size
    fig, ax = plt.subplots(figsize=(20, 12))

    # Plot only odd-numbered agent curves
    for i in range(1, 20, 2):  # Step by 2 to get odd numbers
        agent_col_name = f'Agent {i}'
        if agent_col_name in new_df.columns:
            plt.plot(new_df['Step'], new_df[agent_col_name], 
                    label=agent_col_name, alpha=0.5, linewidth=3)

    # Plot average reward curve
    plt.plot(new_df['Step'], new_df['Average Reward'], 
             label='Average', linewidth=4, color='black')

    # Set plot properties
    plt.xlabel('Time slots (millions)', fontsize=40)
    plt.ylabel('Average episode reward', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=32)
    
    # Configure legend to be inside the plot at the bottom
    plt.legend(fontsize=30, ncol=3, loc='lower center',  # Changed ncol to 3 since we have fewer lines
              bbox_to_anchor=(0.6, 0.15))
    
    plt.grid(True)
    plt.xlim(-0.05e6, 1.0e6)

    # Format x-axis to show values in millions
    def millions_formatter(x, pos):
        return f'{x/1e6:.1f}'
    ax.xaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))

    # Adjust layout
    plt.tight_layout()

    # Save plots
    plt.savefig('./eval_reward_3_57links_separate_with_average.png', 
                format='png', dpi=1600, bbox_inches='tight')
    plt.savefig('./eval_reward_3_57links_separate_with_average.eps', 
                format='eps', dpi=500, bbox_inches='tight')
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    plot_agent_rewards()