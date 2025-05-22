import torch
import torch.nn as nn
import math

class ECALayer(nn.Module):
    """Efficient Channel Attention module."""
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature descriptor via global average pooling
        y = self.avg_pool(x)
        # Reshape for 1D convolution: [B, C, 1, 1] -> [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)
        # Apply 1D convolution
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        # Apply sigmoid to get attention weights
        y = self.sigmoid(y)
        # Scale input features
        return x * y.expand_as(x)