"""
This module provides functionality for testing trained U-Net models for eddy detection.
It handles model loading, inference, and result visualization for both SSH-only and 
multi-variable (SSH, SST, Chlorophyll) inputs.
"""

import os.path
import pickle
import torch
import numpy as np
import yaml
from os.path import join
from models.Models2D import UNet
from viz_utils import visualize_prediction, visualize_inputs

def load_model(model_path: str, input_type: str = 'ssh_sst_chlora', 
               days_before: int = 2, days_after: int = 2) -> UNet:
    """
    Load a trained UNet model from a checkpoint file.
    
    Args:
        model_path: Path to the model weights file
        input_type: Type of input data ('ssh_sst_chlora' or 'only_ssh')
        days_before: Number of previous days to consider
        days_after: Number of following days to consider
    
    Returns:
        model: Loaded and configured UNet model in eval mode
    """
    # Calculate input channels based on data type and temporal window
    tot_inputs = 1 if input_type == 'only_ssh' else 3
    count_channels = tot_inputs + tot_inputs*(days_before+days_after)
        
    # Initialize and load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(num_levels=4, cnn_per_level=2, input_channels=count_channels,
                 output_channels=1, start_filters=32, kernel_size=3).to(device)
    
    # Add map_location to handle device mapping
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path} successfully")
    return model

def test_model(model: UNet, input_data: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """
    Run inference on input data using the provided model.
    
    Args:
        model: Trained UNet model
        input_data: Input data array
        device: Computing device ('cpu' or 'cuda')
    
    Returns:
        numpy array: Model predictions
    """
    model = model.to(device)
    with torch.no_grad():
        inputs = torch.from_numpy(input_data).float().to(device)
        outputs = model(inputs)
    return outputs.cpu().numpy()

def load_config(config_path: str = 'config.yml') -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    required_params = ['inputs', 'days_before', 'lcv_length', 'test_id']
    for param in required_params:
        if param not in config['model']:
            raise ValueError(f"Missing required parameter in config: {param}")
    
    return config['model']

def main():
    """Main function to demonstrate model testing workflow."""
    # Load configuration
    config = load_config()
    
    # Extract parameters
    inputs = config['inputs']
    days_before = config['days_before']
    lcv_length = config['lcv_length']
    test_id = config['test_id']
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load test data
    input_data_file = f"input_data/test_data_prev_days_{days_before}_days_after_{lcv_length}_2010_{inputs}.pkl"
    with open(input_data_file, 'rb') as f:
        input_data = pickle.load(f)
    
    # Setup model paths
    weights_file = f"model_weights_DaysBefore_{days_before:02d}_DaysAfter_{lcv_length:02d}_DaysCoherent_{lcv_length:02d}.pth"
    weights_folder = f"DaysBefore_{days_before:02d}_DaysAfter_{lcv_length:02d}_DaysCoherent_{lcv_length:02d}"
    model_weights_path = join(
        "./model_weights",
        "EddyDetection_ALL_1998-2022_gaps_filled_submean_sst_chlora" if inputs == 'ssh_sst_chlora' else "EddyDetection_ALL_1993-2022_gaps_filled_submean_only_ssh",
        weights_folder,
        weights_file
    )
    
    # Visualize inputs
    output_file = join(config['output_folder'], f"inputs_{inputs}_DaysBefore_{days_before:02d}_DaysAfter_{lcv_length:02d}_DaysCoherent_{lcv_length:02d}.png")
    visualize_inputs(input_data, inputs, days_before, lcv_length, output_file)
    
    # Verify weights exist
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at: {model_weights_path}")
    
    # Load and test model
    model = load_model(model_weights_path, input_type=inputs, 
                      days_before=days_before, days_after=lcv_length)
    
    # Get test sample
    input_data_cur_date = input_data[test_id]['data'][0]
    target = input_data[test_id]['data'][1]
    
    # Make prediction
    prediction = test_model(model, input_data_cur_date[None, ...], device)
    prediction = prediction.squeeze()
    target = target.squeeze()
    
    # Visualize results
    output_file = join(config['output_folder'], f"prediction_{inputs}_DaysBefore_{days_before:02d}_DaysAfter_{lcv_length:02d}_DaysCoherent_{lcv_length:02d}.png")
    visualize_prediction(input_data_cur_date[days_before], prediction, target, output_file)

if __name__ == "__main__":
    main()