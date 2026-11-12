"""
run_comparison.py
=================
Comprehensive Comparative Benchmark & Experimental Suite.

Executes:
1. Base Paper Pipeline: Dual Partition + Recursive Filter + Base 3D-CNN.
2. Our Enhanced Pipeline: Differentiable Band Selector + Anti-Collinearity Loss +
                          Adaptive Guided Filter + Res-SSAN Backbone.
3. Spatial Block Cross-Validation vs. Random Pixel Split (Revealing Data Leakage).
4. Generates quantitative comparison tables (OA, AA, Kappa, MSD, Entropy, Contiguity).
5. Produces publication-ready figures in figures/.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_manager import load_dataset, SpatialSplitter, HSIDataset
from src.baseline.dual_partition import BaseDualPartitionSelector
from src.baseline.recursive_filter import DomainTransformFilter
from src.baseline.base_3d_cnn import Base3DCNN

from src.models.spectral_spatial_backbone import EnhancedHSIClassifier
from src.models.anti_collinearity import AntiCollinearityLoss, SpectralDiversityLoss
from src.utils.metrics import compute_classification_metrics, evaluate_band_redundancy
from src.utils.visualization import (
    plot_band_selection_comparison,
    plot_correlation_matrices,
    plot_classification_maps
)


def train_baseline_pipeline(cube, gt, train_idx, test_idx, target_bands=15, epochs=12, batch_size=64, lr=1e-3, device='cpu'):
    print("\n" + "="*60)
    print(">>> RUNNING BASE PAPER PIPELINE (Kumar et al., 2024)")
    print("="*60)

    # 1. Band Selection via Dual Partitioning & Wavelet Entropy
    t0 = time.time()
    selector = BaseDualPartitionSelector(target_bands=target_bands)
    selected_bands = selector.select_bands(cube)
    t_select = time.time() - t0
    print(f"[*] Base Paper Selected {len(selected_bands)} Bands: {selected_bands}")

    # 2. Bi-hemispherical Reflectance Recovery (Recursive Filtering)
    t0 = time.time()
    cube_sub = cube[:, :, selected_bands]
    dt_filter = DomainTransformFilter()
    filtered_cube = dt_filter.filter_cube(cube_sub)
    t_filter = time.time() - t0
    print(f"[*] Applied 1D Recursive Domain Filter ({t_filter:.2f}s)")

    # 3. Datasets and Loaders
    num_classes = len(np.unique(gt[gt > 0]))
    train_dataset = HSIDataset(filtered_cube, gt, patch_size=7, pixel_indices=train_idx)
    test_dataset = HSIDataset(filtered_cube, gt, patch_size=7, pixel_indices=test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Model Training
    model = Base3DCNN(in_bands=len(selected_bands), patch_size=7, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"[*] Training Base 3D-CNN for {epochs} epochs...")
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (ep + 1) % 4 == 0 or ep == epochs - 1:
            print(f"    Epoch [{ep+1}/{epochs}] - Loss: {total_loss / len(train_loader):.4f}")

    # 5. Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for patches, labels in test_loader:
            patches = patches.to(device)
            logits = model(patches)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    metrics = compute_classification_metrics(all_targets, all_preds, num_classes=num_classes)
    redundancy = evaluate_band_redundancy(cube, selected_bands)

    return {
        "metrics": metrics,
        "redundancy": redundancy,
        "selected_bands": selected_bands,
        "preds": all_preds,
        "targets": all_targets,
        "time_select": t_select,
        "model": model,
        "filtered_cube": filtered_cube
    }


def train_enhanced_pipeline(cube, gt, train_idx, test_idx, target_bands=15, epochs=15, batch_size=64, lr=2e-3, device='cpu'):
    print("\n" + "="*60)
    print(">>> RUNNING OUR ENHANCED PIPELINE (Differentiable + Anti-Collinear + Res-SSAN)")
    print("="*60)

    H, W, total_bands = cube.shape
    num_classes = len(np.unique(gt[gt > 0]))

    # Compute correlation matrix for anti-collinearity repulsion loss
    flat_bands = cube.reshape(-1, total_bands)
    norm_bands = (flat_bands - flat_bands.mean(axis=0)) / (flat_bands.std(axis=0) + 1e-8)
    corr_matrix = np.dot(norm_bands.T, norm_bands) / (H * W)

    # Instantiate end-to-end model
    model = EnhancedHSIClassifier(
        total_bands=total_bands,
        target_bands=target_bands,
        patch_size=7,
        num_classes=num_classes
    ).to(device)

    # Loss components
    ce_loss_fn = nn.CrossEntropyLoss()
    repulse_loss_fn = AntiCollinearityLoss(corr_matrix, sigma_dist=5.0).to(device)
    diversity_loss_fn = SpectralDiversityLoss(cube).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Full cube loaders (raw all bands are fed; selector dynamically activates top-p)
    train_dataset = HSIDataset(cube, gt, patch_size=7, pixel_indices=train_idx)
    test_dataset = HSIDataset(cube, gt, patch_size=7, pixel_indices=test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"[*] Training End-to-End Res-SSAN for {epochs} epochs...")
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, weights = model(patches)

            # Combined multi-task objective: classification guided with spectral decorrelation
            loss_ce = ce_loss_fn(logits, labels)
            # Row entropy penalty (encourages sharp one-hot band selection)
            loss_sparse = 0.02 * torch.sum(weights * torch.log(weights + 1e-10))
            loss_repulse = 0.15 * repulse_loss_fn(weights)
            loss_div = 0.05 * diversity_loss_fn(weights)

            loss = loss_ce + loss_sparse + loss_repulse + loss_div
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.band_selector.step_temperature()
        scheduler.step()

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"    Epoch [{ep+1}/{epochs}] - Loss: {total_loss / len(train_loader):.4f} (Temp: {model.band_selector.temperature:.2f})")

    # Final discrete band selection
    selected_bands = model.band_selector.get_topk_indices()
    print(f"[*] Our Model Discovered {len(selected_bands)} Optimal Bands: {selected_bands}")

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for patches, labels in test_loader:
            patches = patches.to(device)
            logits, _ = model(patches)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    metrics = compute_classification_metrics(all_targets, all_preds, num_classes=num_classes)
    redundancy = evaluate_band_redundancy(cube, selected_bands)

    return {
        "metrics": metrics,
        "redundancy": redundancy,
        "selected_bands": selected_bands,
        "preds": all_preds,
        "targets": all_targets,
        "model": model
    }


def generate_full_prediction_map(model, cube, gt, patch_size=7, is_baseline=False, filtered_cube=None, device='cpu'):
    """Generates 2D prediction map for visualization."""
    H, W = gt.shape
    pred_map = np.zeros((H, W), dtype=np.int32)
    coords = [(r, c) for r in range(H) for c in range(W) if gt[r, c] > 0]
    src_cube = filtered_cube if is_baseline else cube

    eval_dataset = HSIDataset(src_cube, gt, patch_size=patch_size, pixel_indices=coords)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for patches, _ in eval_loader:
            patches = patches.to(device)
            if is_baseline:
                logits = model(patches)
            else:
                logits, _ = model(patches)
            preds = torch.argmax(logits, dim=1).cpu().numpy() + 1
            all_preds.extend(preds)

    for idx, (r, c) in enumerate(coords):
        pred_map[r, c] = all_preds[idx]

    return pred_map


def main():
    parser = argparse.ArgumentParser(description="Hyperspectral Band Selection Benchmark")
    parser.add_argument("--dataset", type=str, default="IndianPines", choices=["IndianPines", "Salinas", "KSC"])
    parser.add_argument("--bands", type=int, default=15, help="Number of bands to select")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--split_mode", type=str, default="spatial_block", choices=["spatial_block", "random"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--fig_dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Benchmark on {args.dataset} (Device: {device}, Split: {args.split_mode})")

    # 1. Load Dataset
    cube, gt = load_dataset(args.dataset)
    train_idx, test_idx = SpatialSplitter.split(gt, train_ratio=0.25, mode=args.split_mode)
    print(f"[*] Total Labeled Samples: {len(train_idx) + len(test_idx)} | Train: {len(train_idx)}, Test: {len(test_idx)}")

    # 2. Run Base Paper Model
    base_res = train_baseline_pipeline(
        cube, gt, train_idx, test_idx,
        target_bands=args.bands,
        epochs=args.epochs,
        device=device
    )

    # 3. Run Our Enhanced Model
    our_res = train_enhanced_pipeline(
        cube, gt, train_idx, test_idx,
        target_bands=args.bands,
        epochs=args.epochs + 2,
        device=device
    )

    # 4. Comparative Metrics Summary
    print("\n" + "="*70)
    print(f"COMPREHENSIVE BENCHMARK RESULTS ON {args.dataset.upper()} ({args.split_mode.upper()} SPLIT)")
    print("="*70)
    print(f"{'Metric':<30} | {'Base Paper (Kumar et al.)':<20} | {'Our Enhanced Model':<20}")
    print("-" * 75)
    print(f"{'Overall Accuracy (OA)':<30} | {base_res['metrics']['OA']:>18.2f}% | {our_res['metrics']['OA']:>18.2f}%")
    print(f"{'Average Accuracy (AA)':<30} | {base_res['metrics']['AA']:>18.2f}% | {our_res['metrics']['AA']:>18.2f}%")
    print(f"{'Cohen Kappa (Kappa)':<30} | {base_res['metrics']['Kappa']:>18.2f}% | {our_res['metrics']['Kappa']:>18.2f}%")
    print(f"{'Mean Spectral Divergence (MSD)':<30} | {base_res['redundancy']['MSD']:>18.2f}  | {our_res['redundancy']['MSD']:>18.2f} ")
    print(f"{'Mean Entropy':<30} | {base_res['redundancy']['MeanEntropy']:>18.2f}  | {our_res['redundancy']['MeanEntropy']:>18.2f} ")
    print(f"{'Pairwise Absolute Correlation':<30} | {base_res['redundancy']['MeanAbsCorrelation']:>18.4f}  | {our_res['redundancy']['MeanAbsCorrelation']:>18.4f} ")
    print(f"{'Contiguous Band Pairs (Bug)':<30} | {base_res['redundancy']['ContiguousPairs']:>18d}  | {our_res['redundancy']['ContiguousPairs']:>18d} ")
    print("="*75)

    # 5. Generate Visualizations
    fig_band_path = os.path.join(args.fig_dir, f"{args.dataset.lower()}_band_selection.png")
    plot_band_selection_comparison(
        cube.shape[2],
        base_res['selected_bands'],
        our_res['selected_bands'],
        dataset_name=args.dataset,
        save_path=fig_band_path
    )

    fig_corr_path = os.path.join(args.fig_dir, f"{args.dataset.lower()}_correlation_heatmaps.png")
    plot_correlation_matrices(
        cube,
        base_res['selected_bands'],
        our_res['selected_bands'],
        save_path=fig_corr_path
    )

    base_map = generate_full_prediction_map(
        base_res['model'], cube, gt,
        is_baseline=True,
        filtered_cube=base_res['filtered_cube'],
        device=device
    )
    our_map = generate_full_prediction_map(
        our_res['model'], cube, gt,
        is_baseline=False,
        device=device
    )

    fig_map_path = os.path.join(args.fig_dir, f"{args.dataset.lower()}_classification_maps.png")
    plot_classification_maps(gt, base_map, our_map, dataset_name=args.dataset, save_path=fig_map_path)

    # 6. Save JSON Summary
    summary_data = {
        "dataset": args.dataset,
        "split_mode": args.split_mode,
        "num_bands": args.bands,
        "baseline": {
            "OA": base_res['metrics']['OA'],
            "AA": base_res['metrics']['AA'],
            "Kappa": base_res['metrics']['Kappa'],
            "MSD": base_res['redundancy']['MSD'],
            "MeanEntropy": base_res['redundancy']['MeanEntropy'],
            "MeanAbsCorrelation": base_res['redundancy']['MeanAbsCorrelation'],
            "ContiguousPairs": base_res['redundancy']['ContiguousPairs'],
            "SelectedBands": base_res['selected_bands']
        },
        "enhanced": {
            "OA": our_res['metrics']['OA'],
            "AA": our_res['metrics']['AA'],
            "Kappa": our_res['metrics']['Kappa'],
            "MSD": our_res['redundancy']['MSD'],
            "MeanEntropy": our_res['redundancy']['MeanEntropy'],
            "MeanAbsCorrelation": our_res['redundancy']['MeanAbsCorrelation'],
            "ContiguousPairs": our_res['redundancy']['ContiguousPairs'],
            "SelectedBands": our_res['selected_bands']
        }
    }
    json_path = os.path.join(args.output_dir, f"{args.dataset.lower()}_benchmark_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"[+] Benchmark metrics saved to: {json_path}")


if __name__ == "__main__":
    main()
