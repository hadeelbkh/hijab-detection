# model.py
import torch.nn as nn
from torchvision import models
from src.config import Config

def build_model():
    # Load pre-trained model
    if Config.MODEL_NAME == "resnet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        # Replace final layer with dropout and binary output
        model.fc = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT), # Important to reduce overfitting
            nn.Linear(num_ftrs, 1)  # Binary output
        )
    # We can try more than one model
    # and choose the best one based on the results
    elif Config.MODEL_NAME == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT), 
            nn.Linear(num_ftrs, 1)
        )
    elif Config.MODEL_NAME == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT),
            nn.Linear(num_ftrs, 1)
        )
    else:
        raise ValueError(f"Unsupported model: {Config.MODEL_NAME}")

    # Move the model to a specified device
    model = model.to(Config.DEVICE)
    return model