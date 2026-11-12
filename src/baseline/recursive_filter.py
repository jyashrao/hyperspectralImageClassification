"""
recursive_filter.py
===================
Bi-Hemispherical Reflectance Recovery via 1D Recursive Domain-Transform Filtering.
Implements Equations (11)-(14) of Kumar et al. (2024).

The filter uses an isometric distance transform delta = integral(1 + (sp/sq) * |H'(x)| dx)
to perform edge-preserving smoothing across horizontal and vertical sweeps:
    H_bss[k] = (1 - b^delta) * H_bs[k] + a^delta * H_bss[k - 1]
"""

import numpy as np


class DomainTransformFilter:
    def __init__(self, sigma_s=5.0, sigma_r=0.2, num_iterations=2):
        self.sigma_s = sigma_s  # Spatial bandwidth
        self.sigma_r = sigma_r  # Range bandwidth
        self.num_iterations = num_iterations

    def _filter_1d_horizontal(self, img, d_img):
        """
        Recursive 1D horizontal pass: forward and backward.
        img: (H, W, C)
        d_img: (H, W) spatial derivative/distance
        """
        H, W, C = img.shape
        out = img.copy()

        # Compute feedback factor: a = exp(-sqrt(2) / sigma_s)
        a = np.exp(-np.sqrt(2.0) / self.sigma_s)

        # Forward pass (left to right)
        for r in range(H):
            for c in range(1, W):
                delta = d_img[r, c]
                weight = a ** delta
                out[r, c, :] = (1.0 - weight) * img[r, c, :] + weight * out[r, c - 1, :]

        # Backward pass (right to left)
        for r in range(H):
            for c in range(W - 2, -1, -1):
                delta = d_img[r, c]
                weight = a ** delta
                out[r, c, :] = (1.0 - weight) * out[r, c, :] + weight * out[r, c + 1, :]

        return out

    def filter_cube(self, cube):
        """
        Applies 2D domain transform filtering via alternating horizontal and vertical 1D passes.
        cube: (H, W, p)
        """
        H, W, p = cube.shape
        filtered = cube.copy().astype(np.float32)

        for _ in range(self.num_iterations):
            # 1. Horizontal gradient & domain distance
            dH = np.zeros((H, W), dtype=np.float32)
            dH[:, 1:] = 1.0 + (self.sigma_s / self.sigma_r) * np.sum(np.abs(np.diff(filtered, axis=1)), axis=-1)
            filtered = self._filter_1d_horizontal(filtered, dH)

            # 2. Vertical gradient & domain distance (transpose trick)
            filtered_T = np.transpose(filtered, (1, 0, 2))
            dV = np.zeros((W, H), dtype=np.float32)
            dV[:, 1:] = 1.0 + (self.sigma_s / self.sigma_r) * np.sum(np.abs(np.diff(filtered_T, axis=1)), axis=-1)
            filtered_T = self._filter_1d_horizontal(filtered_T, dV)
            filtered = np.transpose(filtered_T, (1, 0, 2))

        return filtered
