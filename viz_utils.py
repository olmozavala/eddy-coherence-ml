"""
This module provides visualization utilities for displaying model inputs,
predictions, and ground truth data for eddy detection tasks.
"""

import matplotlib.pyplot as plt
import cmocean
import numpy as np
from matplotlib.colors import LogNorm
from typing import List, Dict, Any
from os.path import join
import os
def visualize_prediction(input_data_cur_date: np.ndarray,
                        prediction: np.ndarray,
                        target: np.ndarray,
                        output_file: str= join("outputs", "prediction.png")) -> None:
    """
    Visualize model prediction alongside ground truth.

    Args:
        input_data_cur_date: SSH data for current timestep
        prediction: Model's prediction
        target: Ground truth data
        output_file: File to save visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot SSH with prediction overlay
    # Plot SSH with ground truth overlay
    im1 = ax1.imshow(input_data_cur_date, cmap=cmocean.cm.balance, origin='lower')
    ax1.set_title('SSH with Ground Truth')
    # Create mask for ground truth
    mask = target > 0.5
    ax1.imshow(np.ma.masked_where(~mask, target), cmap='Greys', alpha=0.7, origin='lower')

    # Plot SSH with prediction overlay
    im2 = ax2.imshow(input_data_cur_date, cmap=cmocean.cm.balance, origin='lower')
    ax2.set_title('SSH with Model Prediction')
    # Create mask for prediction
    mask = prediction > 0.5
    ax2.imshow(np.ma.masked_where(~mask, prediction), cmap='Greys', alpha=0.7, origin='lower')

    plt.tight_layout()
    # Verify the output folder exists
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    plt.savefig(output_file)
    plt.show()

def visualize_inputs(data: List[Dict[str, Any]],
                    inputs_str: str,
                    days_before: int,
                    lcv_length: int,
                    output_file: str= join("outputs", "inputs.png")) -> None:
    """
    Visualize all input variables across the temporal window.

    Args:
        data: List of dictionaries containing input data
        inputs_str: Type of inputs ('ssh_sst_chlora' or 'only_ssh')
        days_before: Number of previous days
        lcv_length: Length of coherent structure
        output_file: File to save visualization
    """
    # Calculate total number of input channels
    total_inputs = (days_before + lcv_length + 1) * (3 if inputs_str == 'ssh_sst_chlora' else 1)

    # Create subplot grid
    num_rows = (total_inputs + 3) // 4
    fig, axs = plt.subplots(num_rows, min(total_inputs, 4), figsize=(20, 3*num_rows))
    axs = axs.reshape(num_rows, -1) if num_rows == 1 else axs

    # Plot each input channel
    inputs_per_var = days_before + lcv_length + 1
    for i in range(total_inputs):
        row, col = i // 4, i % 4
        ax = axs[row, col]

        if i < inputs_per_var:  # SSH
            im = ax.imshow(data[0]['data'][0][i], cmap=cmocean.cm.balance, origin='lower')
            ax.set_title(f'SSH Day {i-days_before}')
        elif i < inputs_per_var*2:  # SST
            im = ax.imshow(data[0]['data'][0][i], cmap=cmocean.cm.thermal, origin='lower')
            ax.set_title(f'SST Day {i-inputs_per_var-days_before}')
        else:  # Chlorophyll
            im = ax.imshow(data[0]['data'][0][i], cmap=cmocean.cm.algae, origin='lower')
            ax.set_title(f'Chlor-a Day {i-2*inputs_per_var-days_before}')

        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)

    # Main title
    plt.suptitle(f"Inputs for {inputs_str} with {days_before} days before and {lcv_length} days after", fontsize=16)
    plt.tight_layout()
    # Verify the output folder exists
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    plt.savefig(output_file)
    plt.show()