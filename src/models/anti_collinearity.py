"""
anti_collinearity.py
====================
Anti-Collinearity Repulsion & Spectral Diversity Loss Modules.

Formulation:
Prevents two selector rows from choosing spectrally adjacent and correlated bands:
    L_repulse = sum_{k != m} sum_{i, j} W_{k, i} * W_{m, j} * |C_{i, j}| * exp(-|i - j|^2 / (2 * sigma^2))
"""

import numpy as np
import torch
import torch.nn as nn


class AntiCollinearityLoss(nn.Module):
    def __init__(self, corr_matrix, sigma_dist=4.0):
        super(AntiCollinearityLoss, self).__init__()
        if isinstance(corr_matrix, np.ndarray):
            corr_tensor = torch.from_numpy(corr_matrix).float()
        else:
            corr_tensor = corr_matrix.float()

        self.register_buffer("abs_corr", torch.abs(corr_tensor))
        d = corr_tensor.size(0)

        # Distance matrix: |i - j|
        indices = torch.arange(d).unsqueeze(1)
        dist_matrix = torch.abs(indices - indices.t()).float()
        proximity = torch.exp(-(dist_matrix ** 2) / (2.0 * (sigma_dist ** 2)))
        self.register_buffer("proximity", proximity)

    def forward(self, weights):
        """
        weights: (target_bands, total_bands) or (total_bands,)
        """
        if weights.dim() == 1:
            w = weights.unsqueeze(1)
            outer_w = torch.mm(w, w.t())
            repulsion = outer_w * self.abs_corr * self.proximity
            return torch.sum(repulsion)

        # weights: (K, d)
        # Compute pairwise cross-row repulsion
        K, d = weights.size()
        # Affinity matrix A = abs_corr * proximity in R^(d, d)
        affinity = self.abs_corr * self.proximity
        
        # Row-to-affinity: W * A in R^(K, d)
        WA = torch.matmul(weights, affinity)
        # Cross-row dot product: WA * W^T in R^(K, K)
        row_repulsion = torch.matmul(WA, weights.t())
        
        # Zero out diagonal (self-row affinity is ignored; only cross-row is penalized)
        row_repulsion.fill_diagonal_(0.0)
        return torch.sum(row_repulsion) / (K * (K - 1) + 1e-8)


class SpectralDiversityLoss(nn.Module):
    """Encourages divergence across chosen bands."""
    def __init__(self, cube):
        super(SpectralDiversityLoss, self).__init__()
        H, W, d = cube.shape
        flat = cube.reshape(-1, d)
        flat = flat / (np.sum(flat, axis=0, keepdims=True) + 1e-8)

        dist_matrix = np.zeros((d, d), dtype=np.float32)
        for i in range(d):
            dist_matrix[i, :] = np.sum(np.abs(flat[:, [i]] - flat), axis=0)

        dist_matrix = dist_matrix / (dist_matrix.max() + 1e-8)
        self.register_buffer("div_matrix", torch.from_numpy(dist_matrix).float())

    def forward(self, weights):
        if weights.dim() == 2:
            # Flatten to marginal distribution
            weights = torch.mean(weights, dim=0)
        w = weights.unsqueeze(1)
        outer_w = torch.mm(w, w.t())
        divergence = torch.sum(outer_w * self.div_matrix)
        return -divergence
