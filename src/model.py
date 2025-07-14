# model.py
import torch.nn as nn
from torchvision import models
from src.config import Config
from torchvision import models
from src.eca_module import ECALayer
from torchvision.models import (
    ResNet50_Weights, 
    MobileNet_V2_Weights, 
    EfficientNet_B0_Weights,
    DenseNet121_Weights
)

def build_model():
    # Load pre-trained model
    if Config.MODEL_NAME == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        # Replace final layer with dropout and binary output
        model.fc = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT), # Important to reduce overfitting
            nn.Linear(num_ftrs, 1)  # Binary output
        )
    # We can try more than one model
    # and choose the best one based on the results
    elif Config.MODEL_NAME == "mobilenetv2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT), 
            nn.Linear(num_ftrs, 1)
        )
    elif Config.MODEL_NAME == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT),
            nn.Linear(num_ftrs, 1)
        )
    elif Config.MODEL_NAME == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features

        # نبني features جديدة ونضيف ECA بعد كل denseblock
        new_features = nn.Sequential()
        for name, layer in model.features._modules.items():
            new_features.add_module(name, layer)
            if "denseblock" in name:
                last_dense = list(layer.children())[-1]
                out_channels = last_dense.norm2.num_features
                new_features.add_module(f"eca_after_{name}", ECALayer(channel=out_channels))

        model.features = new_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT),
            nn.Linear(num_ftrs, 1)
        )
    else:
        raise ValueError(f"Unsupported model: {Config.MODEL_NAME}")

    # Move the model to a specified device
    model = model.to(Config.DEVICE)
    return model