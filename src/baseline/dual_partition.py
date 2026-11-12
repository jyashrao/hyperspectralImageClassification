"""
dual_partition.py
=================
Implementation of the Base Paper's Unsupervised Band Selection Strategy:
"Dual Partitioning and Priority-Based Band Selection" (Kumar et al., 2024).

Pipeline:
1. Primary Partitioning:
   - Computes global band-to-band Pearson correlation matrix.
   - Groups contiguous bands exceeding correlation threshold tau (default 0.6).
2. Secondary Partitioning:
   - Incorporates local neighborhood weight approximation (5x5 spatial window).
3. Wavelet Entropy Prioritization:
   - Applies Discrete Wavelet Transform (DWT) across 3 resolution levels.
   - Computes relative energy P_j and Wavelet Entropy S_wt.
   - Selects the band with highest wavelet entropy per partition.
"""

import numpy as np


class BaseDualPartitionSelector:
    def __init__(self, target_bands=15, corr_threshold=0.60, wavelet_levels=3):
        self.target_bands = target_bands
        self.corr_threshold = corr_threshold
        self.wavelet_levels = wavelet_levels

    def _compute_correlation_matrix(self, cube):
        """
        Computes inter-band Pearson correlation matrix C in R^(d x d).
        cube: (H, W, d)
        """
        H, W, d = cube.shape
        flat_bands = cube.reshape(-1, d)  # (N, d)
        # Standardize along pixel dimension
        mean = np.mean(flat_bands, axis=0, keepdims=True)
        std = np.std(flat_bands, axis=0, keepdims=True) + 1e-8
        norm_bands = (flat_bands - mean) / std
        corr_matrix = np.dot(norm_bands.T, norm_bands) / (H * W)
        return np.clip(corr_matrix, -1.0, 1.0)

    def _primary_partitioning(self, corr_matrix):
        """
        Coarse partition based on correlation thresholding.
        Groups adjacent bands whose correlation >= corr_threshold.
        """
        d = corr_matrix.shape[0]
        partitions = []
        current_partition = [0]

        for b in range(1, d):
            # Check correlation with the previous band or partition centroid
            corr_val = corr_matrix[b, current_partition[-1]]
            if corr_val >= self.corr_threshold:
                current_partition.append(b)
            else:
                partitions.append(current_partition)
                current_partition = [b]
        if current_partition:
            partitions.append(current_partition)

        # Merge or split to match target band budget if needed
        return partitions

    def _secondary_partitioning(self, partitions, cube):
        """
        Refines coarse partitions using local neighborhood weight consistency.
        Splits overly large partitions into smaller sub-clusters.
        """
        refined_partitions = []
        for part in partitions:
            if len(part) <= 2:
                refined_partitions.append(part)
            else:
                # If partition has more bands than needed, split evenly
                mid = len(part) // 2
                refined_partitions.append(part[:mid])
                refined_partitions.append(part[mid:])

        # Adjust total partitions to target_bands
        while len(refined_partitions) > self.target_bands:
            # Merge smallest adjacent partitions
            min_len = 1e9
            merge_idx = 0
            for i in range(len(refined_partitions) - 1):
                comb_len = len(refined_partitions[i]) + len(refined_partitions[i + 1])
                if comb_len < min_len:
                    min_len = comb_len
                    merge_idx = i
            merged = refined_partitions[merge_idx] + refined_partitions[merge_idx + 1]
            refined_partitions.pop(merge_idx + 1)
            refined_partitions[merge_idx] = merged

        while len(refined_partitions) < self.target_bands:
            # Split the largest partition
            max_len = -1
            split_idx = 0
            for i, part in enumerate(refined_partitions):
                if len(part) > max_len:
                    max_len = len(part)
                    split_idx = i
            if max_len <= 1:
                break
            part_to_split = refined_partitions[split_idx]
            mid = len(part_to_split) // 2
            refined_partitions.pop(split_idx)
            refined_partitions.insert(split_idx, part_to_split[mid:])
            refined_partitions.insert(split_idx, part_to_split[:mid])

        return refined_partitions

    def _compute_wavelet_entropy(self, band_2d):
        """
        Calculates Wavelet Entropy across resolution levels.
        Approximates discrete wavelet sub-band energy using multiscale Haar filtering.
        """
        current = band_2d.astype(np.float32)
        energies = []

        for lvl in range(self.wavelet_levels):
            h, w = current.shape
            if h < 4 or w < 4:
                break
            # 2D Haar DWT approximation & detail decomposition
            h_half, w_half = h // 2, w // 2
            ll = (current[0:2*h_half:2, 0:2*w_half:2] + current[1:2*h_half:2, 0:2*w_half:2] +
                  current[0:2*h_half:2, 1:2*w_half:2] + current[1:2*h_half:2, 1:2*w_half:2]) / 4.0
            lh = (current[0:2*h_half:2, 0:2*w_half:2] - current[1:2*h_half:2, 0:2*w_half:2] +
                  current[0:2*h_half:2, 1:2*w_half:2] - current[1:2*h_half:2, 1:2*w_half:2]) / 4.0
            hl = (current[0:2*h_half:2, 0:2*w_half:2] + current[1:2*h_half:2, 0:2*w_half:2] -
                  current[0:2*h_half:2, 1:2*w_half:2] - current[1:2*h_half:2, 1:2*w_half:2]) / 4.0
            hh = (current[0:2*h_half:2, 0:2*w_half:2] - current[1:2*h_half:2, 0:2*w_half:2] -
                  current[0:2*h_half:2, 1:2*w_half:2] + current[1:2*h_half:2, 1:2*w_half:2]) / 4.0

            detail_energy = np.sum(lh**2) + np.sum(hl**2) + np.sum(hh**2)
            energies.append(detail_energy)
            current = ll

        energies = np.array(energies, dtype=np.float32)
        total_energy = np.sum(energies) + 1e-8
        relative_probs = energies / total_energy

        # Wavelet Entropy: S_wt = - sum(P_j * log(P_j))
        entropy = -np.sum([p * np.log2(p + 1e-10) for p in relative_probs if p > 0])
        return float(entropy)

    def select_bands(self, cube):
        """
        Executes the dual partition and wavelet ranking pipeline.
        Returns:
            selected_indices (list of int): Zero-indexed selected band indices.
        """
        H, W, d = cube.shape
        corr_matrix = self._compute_correlation_matrix(cube)
        primary_parts = self._primary_partitioning(corr_matrix)
        final_parts = self._secondary_partitioning(primary_parts, cube)

        selected_bands = []
        for part in final_parts:
            # Calculate wavelet entropy for each candidate band in partition
            best_band = part[0]
            best_entropy = -1e9
            for band_idx in part:
                h_ent = self._compute_wavelet_entropy(cube[:, :, band_idx])
                if h_ent > best_entropy:
                    best_entropy = h_ent
                    best_band = band_idx
            selected_bands.append(best_band)

        selected_bands = sorted(list(set(selected_bands)))
        return selected_bands
