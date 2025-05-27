# config.py
'''Stores the hyperparameters, paths, and experiment configurations.'''
import torch
from pathlib import Path

class Config:
    # Paths
    ## DATA_DIR = Path('sample/')
    MODEL_NAME = "densenet121"  # We aim to test: resnet50, mobilenetv2, efficientnet_b0, densenet121
    DATA_DIR = Path('/content/drive/MyDrive/hijab-detection/sitr_data/')
    MODEL_PATH = Path(f"/content/drive/MyDrive/hijab-detection/model/tuned-densenet121/")
    NUM_RUNS = 1
    # Training hyperparameters
    BATCH_SIZE = 8  # 32 is too large for Google Colab’s free GPU
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT = 0.5
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")