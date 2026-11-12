"""
base_3d_cnn.py
==============
Implementation of the Base Paper's 3D-CNN Classifier.
Corresponds to Equations (15)-(17) of Kumar et al. (2024).

Architecture:
- Input shape: (Batch_Size, 1, Bands, Height, Width)
- 3D Convolution (captures joint spatio-spectral features) + ReLU
- 3D Max-Pooling (spatial-spectral downsampling)
- Flattening -> Dense Hidden Layer -> Output Logits (Softmax)
"""

import torch
import torch.nn as nn


class Base3DCNN(nn.Module):
    def __init__(self, in_bands=15, patch_size=7, num_classes=16):
        super(Base3DCNN, self).__init__()
        self.in_bands = in_bands
        self.patch_size = patch_size
        self.num_classes = num_classes

        # 3D Convolutional Layer (Eq. 15)
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        self.relu = nn.ReLU(inplace=True)

        # 3D Pooling Layer (downsamples spectral and spatial dimensions)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Dynamically calculate flattened feature vector dimension (FV)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_bands, patch_size, patch_size)
            feat = self.pool(self.relu(self.conv1(dummy)))
            self.flat_dim = feat.view(1, -1).size(1)

        # Fully Connected Layers (Eq. 17)
        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        x: (B, C, H, W) where C is the number of spectral bands.
        Reshaped to (B, 1, C, H, W) for 3D convolution.
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, C, H, W)

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten to FV
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
