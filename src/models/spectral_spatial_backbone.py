"""
spectral_spatial_backbone.py
============================
State-of-the-Art Residual Spectral-Spatial Attention Network (Res-SSAN).
Unifies Differentiable Band Selection, Spatial Guided Filtering, and Deep Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_band_selector import DifferentiableBandSelector
from .adaptive_guided_filter import AdaptiveGuidedFilter


class SpectralAttention(nn.Module):
    """Channel-wise Squeeze-and-Excitation for spectral calibration."""
    def __init__(self, channels, reduction=4):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialResidualBlock(nn.Module):
    """Residual 2D Spatial Conv Block with skip connection."""
    def __init__(self, in_channels, out_channels):
        super(SpatialResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)


class EnhancedHSIClassifier(nn.Module):
    def __init__(self, total_bands=200, target_bands=15, patch_size=7, num_classes=16):
        super(EnhancedHSIClassifier, self).__init__()
        self.total_bands = total_bands
        self.target_bands = target_bands
        self.patch_size = patch_size
        self.num_classes = num_classes

        # 1. End-to-end differentiable band selection
        self.band_selector = DifferentiableBandSelector(
            total_bands=total_bands,
            target_bands=target_bands
        )

        # 2. Learnable Spatial Filtering on selected bands
        self.spatial_filter = AdaptiveGuidedFilter(channels=target_bands)

        # 3. 3D Convolution layer for initial spatio-spectral extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )

        # 4. Dimension reduction from 3D feature map to 2D
        mid_channels = 8 * target_bands
        self.conv2d_reduce = nn.Sequential(
            nn.Conv2d(mid_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 5. Spectral Attention & Deep Spatial Residual Feature Learning
        self.spectral_att = SpectralAttention(64)
        self.res_block1 = SpatialResidualBlock(64, 64)
        self.res_block2 = SpatialResidualBlock(64, 128)

        # 6. Global Pooling & Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: (B, total_bands, H, W)
        """
        B, C, H, W = x.size()

        # Step 1: Differentiable Band Selection
        active_x, selection_weights = self.band_selector(x)  # (B, target_bands, H, W)

        # Step 2: Adaptive Spatial Filtering
        filtered_x = self.spatial_filter(active_x)  # (B, target_bands, H, W)

        # Step 3: 3D Convolution
        x_3d = filtered_x.unsqueeze(1)  # (B, 1, target_bands, H, W)
        feat_3d = self.conv3d(x_3d)    # (B, 8, target_bands, H, W)

        # Step 4: Reshape 3D to 2D
        feat_2d = feat_3d.view(B, -1, H, W)
        feat_2d = self.conv2d_reduce(feat_2d)  # (B, 64, H, W)

        # Step 5: Spectral Attention & Residual Spatial Learning
        feat_att = self.spectral_att(feat_2d)
        res1 = self.res_block1(feat_att)
        res2 = self.res_block2(res1)

        # Step 6: Classification
        pooled = self.global_pool(res2).view(B, -1)
        logits = self.classifier(pooled)

        return logits, selection_weights
