# config.py
'''Stores the hyperparameters, paths, and experiment configurations.'''
import torch
from pathlib import Path

class Config:
    # Paths
    DATA_DIR = Path('sample/')
    ## DATA_DIR = Path('data/')
    MODEL_PATH = Path('model/')
    
    # Training hyperparameters
    BATCH_SIZE = 8  # 32 is too large for Google Colab’s free GPU
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT = 0.5
    
    # Model architecture
    MODEL_NAME = "resnet50"  # We can use: resnet50, mobilenetv2, efficientnet_b0
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")