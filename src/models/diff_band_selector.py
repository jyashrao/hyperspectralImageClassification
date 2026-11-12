"""
diff_band_selector.py
=====================
Task-Aware Differentiable Concrete Band Selector.

Architecture:
- Parameterizes selection via projection matrix logits L in R^(target_bands x total_bands).
- Each row k corresponds to a learnable spectral selector with Gumbel-Softmax relaxation:
      W_k = Softmax((L_k + G_k) / tau)
- Active bands are formed via differentiable projection:
      X_selected = einsum('bchw, kc -> bkhw', X, W)
- At inference, discrete bands are given by argmax_c(L_k).
- Guaranteed stable channel alignment and end-to-end gradient flow.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableBandSelector(nn.Module):
    def __init__(self, total_bands=200, target_bands=15, initial_temp=1.0, min_temp=0.1, anneal_rate=0.92):
        super(DifferentiableBandSelector, self).__init__()
        self.total_bands = total_bands
        self.target_bands = target_bands
        self.temperature = initial_temp
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate

        # Initialize each selector row centered at regular spectral intervals
        logits_init = torch.randn(target_bands, total_bands) * 0.1
        centers = np.linspace(0, total_bands - 1, target_bands).astype(int)
        for k, c in enumerate(centers):
            # Prioritize local neighborhood of center
            logits_init[k, max(0, c - 2):min(total_bands, c + 3)] += 2.0

        self.selector_logits = nn.Parameter(logits_init)

    def step_temperature(self):
        """Anneals temperature toward min_temp over training epochs."""
        self.temperature = max(self.min_temp, self.temperature * self.anneal_rate)

    def get_selection_weights(self):
        """
        Returns W in R^(target_bands, total_bands) representing selection distributions.
        """
        if self.training:
            gumbels = -torch.empty_like(self.selector_logits).exponential_().log()
            noisy_logits = (self.selector_logits + gumbels) / self.temperature
        else:
            noisy_logits = self.selector_logits / max(self.temperature, 0.1)

        weights = F.softmax(noisy_logits, dim=-1)
        return weights

    def get_topk_indices(self):
        """Returns sorted list of discrete chosen band indices (one per row)."""
        with torch.no_grad():
            chosen = torch.argmax(self.selector_logits, dim=-1).cpu().numpy().tolist()
            return sorted(list(set(chosen)))

    def forward(self, x):
        """
        x: (B, total_bands, H, W)
        Returns:
            active_x: (B, target_bands, H, W)
            weights: (target_bands, total_bands)
        """
        weights = self.get_selection_weights()  # (target_bands, total_bands)

        if not self.training:
            # Discrete hard selection at test time
            top_idx = torch.argmax(self.selector_logits, dim=-1)  # (target_bands,)
            active_x = x[:, top_idx, :, :]
        else:
            # Differentiable soft continuous selection: einsum 'bchw, kc -> bkhw'
            active_x = torch.einsum('bchw, kc -> bkhw', x, weights)

        return active_x, weights
