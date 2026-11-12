"""
metrics.py
==========
Standardized Hyperspectral Evaluation Metrics:
- Overall Accuracy (OA)
- Average Accuracy (AA)
- Cohen's Kappa Coefficient (kappa)
- Confusion Matrix
- Mean Spectral Divergence (MSD) via symmetric Kullback-Leibler Divergence (Eq. 18)
- Information Entropy of selected spectral bands
- Contiguity & Collinearity Index
"""

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def compute_classification_metrics(y_true, y_pred, num_classes=None):
    """
    Computes OA, AA, and Kappa from true and predicted class labels.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    # Overall Accuracy
    oa = accuracy_score(y_true, y_pred) * 100.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if num_classes is None:
        num_classes = cm.shape[0]

    # Average Accuracy (per-class recall average)
    per_class_acc = []
    for c in range(cm.shape[0]):
        total_samples = np.sum(cm[c, :])
        if total_samples > 0:
            acc = (cm[c, c] / total_samples) * 100.0
            per_class_acc.append(acc)
    aa = np.mean(per_class_acc) if per_class_acc else 0.0

    # Cohen's Kappa Coefficient
    kappa = cohen_kappa_score(y_true, y_pred) * 100.0

    return {
        "OA": float(oa),
        "AA": float(aa),
        "Kappa": float(kappa),
        "PerClassAcc": [float(a) for a in per_class_acc],
        "ConfusionMatrix": cm
    }


def compute_spectral_entropy(band_2d, num_bins=128):
    """
    Computes Shannon entropy of a single 2D spectral band.
    H = - sum(p * log2(p))
    """
    flat = band_2d.flatten()
    hist, _ = np.histogram(flat, bins=num_bins, density=True)
    hist = hist[hist > 0]
    # Normalize to probability mass
    p = hist / np.sum(hist)
    entropy = -np.sum(p * np.log2(p + 1e-12))
    return float(entropy)


def compute_msd(cube, selected_bands, num_bins=64):
    """
    Computes Mean Spectral Divergence (MSD) as formulated in Eq. (18):
        MSD(S) = (1 / (n * (n - 1))) * sum_{i=1}^n sum_{j=1}^n D_SKL(B_i || B_j)
    where D_SKL is the symmetric Kullback-Leibler divergence between grey histograms.
    Higher MSD indicates greater spectral dissimilarity and lower redundancy.
    """
    n = len(selected_bands)
    if n <= 1:
        return 0.0

    # Compute normalized 1D intensity histograms for all selected bands
    histograms = []
    for b in selected_bands:
        band = cube[:, :, b].flatten()
        h, _ = np.histogram(band, bins=num_bins, range=(0.0, 1.0), density=False)
        p = (h + 1e-6) / np.sum(h + 1e-6)  # Laplace smoothed probability
        histograms.append(p)

    total_div = 0.0
    for i in range(n):
        p_i = histograms[i]
        for j in range(n):
            if i == j:
                continue
            p_j = histograms[j]
            # Symmetrical KL divergence: D_SKL(P || Q) = KL(P || Q) + KL(Q || P)
            kl_ij = np.sum(p_i * np.log((p_i / p_j) + 1e-12))
            kl_ji = np.sum(p_j * np.log((p_j / p_i) + 1e-12))
            total_div += (kl_ij + kl_ji)

    msd = total_div / (n * (n - 1))
    return float(msd)


def evaluate_band_redundancy(cube, selected_bands):
    """
    Analyzes redundancy of selected bands:
    - Average pairwise absolute correlation
    - Count of adjacent/contiguous pairs (|b_i - b_j| == 1)
    - Mean Spectral Divergence (MSD)
    - Mean Entropy
    """
    selected_bands = sorted(selected_bands)
    n = len(selected_bands)
    H, W, _ = cube.shape

    # 1. Contiguous band count
    contiguous_count = 0
    for i in range(n - 1):
        if selected_bands[i + 1] - selected_bands[i] == 1:
            contiguous_count += 1

    # 2. Pairwise correlation
    flat_selected = cube[:, :, selected_bands].reshape(-1, n)
    corr_mat = np.corrcoef(flat_selected, rowvar=False)
    # Extract upper triangular off-diagonal elements
    triu_idx = np.triu_indices(n, k=1)
    mean_abs_corr = np.mean(np.abs(corr_mat[triu_idx])) if len(triu_idx[0]) > 0 else 0.0

    # 3. MSD and Mean Entropy
    msd = compute_msd(cube, selected_bands)
    entropies = [compute_spectral_entropy(cube[:, :, b]) for b in selected_bands]
    mean_entropy = np.mean(entropies)

    return {
        "SelectedBands": selected_bands,
        "ContiguousPairs": int(contiguous_count),
        "MeanAbsCorrelation": float(mean_abs_corr),
        "MSD": float(msd),
        "MeanEntropy": float(mean_entropy)
    }
