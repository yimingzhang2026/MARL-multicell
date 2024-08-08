# Multi-Agent Reinforcement Learning for Multi-Cell Spectrum and Power Allocation

## Important Notice

This code is provided exclusively for the purpose of peer review related to the manuscript submission:

**"Multi-Agent Reinforcement Learning for Multi-Cell Spectrum and Power Allocation"**

**Please do not distribute this code at this time.** The code will be released publicly after the manuscript decision.

## Repository Structure

The repository is organized into four main folders:

1. `8device_conflict_graph_benchmarks`: Contains benchmark algorithms for the conflict graph scenario.
2. `57device_celluar_network_benchmarks`: Contains benchmark algorithms for the cellular network scenario.
3. `MARL_example_cellular`: Contains example of MARL algorithms for the celluar network scenario.
2. `MARL_example_conflict`: Contains example of MARL algorithms for the conflict scenario.

## Generating Simulation Results

### CDF Results (Fig. 7 and Fig. 11)
To generate the Cumulative Distribution Function (CDF) results shown in Fig. 7 and Fig. 11:
1. Navigate to the corresponding directory (`8device_conflict_graph_benchmarks` or `57device_celluar_network_benchmarks`).
2. Run the script: `python generate_cdf.py`

### Packet Delay Results (Fig. 8, Fig. 9, and Fig 10)
To generate the packet delay results shown in Fig. 8, Fig. 9, and Fig 10:
1. Navigate to the corresponding directory.
2. Run the script: `python generate_cluster.py`

## MARL Training Examples

We provide two Multi-Agent Reinforcement Learning (MARL) training examples:

1. Conflict Graph Scenario
2. Cellular Network Scenario

### Pre-trained Models
Pre-trained models are available in the `results` folder of each scenario.

### Training from Scratch
To train the models from scratch:
1. Navigate to the corresponding experiment setting folder.
2. Enter the `train` subfolder.
3. Run the training script: `python train.py`

## Additional Information

- For detailed methodology and results interpretation, please refer to the associated manuscript.
- If you encounter any issues or have questions, please contact the authors through the journal's communication channels.

---

© 2024] [Yiming Zhang/Northwestern University]. All rights reserved.
