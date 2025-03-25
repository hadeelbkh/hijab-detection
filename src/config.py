# config.py
'''Stores the hyperparameters, paths, and experiment configurations.'''
import torch
from pathlib import Path

class Config:
    # Paths
    DATA_DIR = Path('sample/')
    BATCH_SIZE = 4
    ## DATA_DIR = Path('data/')
    ## MODEL_PATH = Path('model/hijab_detector.pth')
    
    # Model architecture
    # MODEL_NAME = "resnet50"  # Options: resnet50, mobilenetv2, efficientnet_b0
    
    # Device
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")