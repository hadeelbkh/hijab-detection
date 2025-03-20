# config.py
'''Stores the hyperparameters, paths, and experiment configurations.'''
import torch
from pathlib import Path

class Config:
    # Paths
    DATA_DIR = Path('data/')
    MODEL_PATH = Path('model/hijab_detector.pth')