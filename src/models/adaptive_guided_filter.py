"""
adaptive_guided_filter.py
=========================
Deep Learnable Bilateral Spatial Guided Filter.

Replaces the base paper's statically tuned 1D recursive domain filter with an adaptive,
gradient-aware spatial filter that dynamically preserves land-cover boundaries.
"""

import torch
import torch.nn as nn


class AdaptiveGuidedFilter(nn.Module):
    def __init__(self, channels=15):
        super(AdaptiveGuidedFilter, self).__init__()
        self.channels = channels

        # Spatial Guidance Map Generator
        self.guidance_net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Multiplicative attenuation weights in [0, 1]
        )

        # Residual spatial refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns:
            filtered_x: (B, C, H, W) with edge-preserved spatial filtering
        """
        # Guidance mask highlights sharp spatial transitions
        guidance = self.guidance_net(x)
        # Bilateral modulation
        smoothed = x * guidance + (1.0 - guidance) * self.refine_conv(x)
        # Residual shortcut connection
        return x + 0.5 * smoothed
