# Next-Generation Hyperspectral Band Selection & Classification Framework

A research-grade PyTorch implementation reproducing and critically enhancing the hyperspectral band selection methodology published in *Scientific Reports* (2024):
> **"A new band selection framework for hyperspectral remote sensing image classification"**  
> *B. L. N. Phaneendra Kumar, Radhesyam Vaddi, Prabukumar Manoharan, L. Agilandeeswari, & V. Sangeetha*  
> *Scientific Reports*, 14:31836, 2024. [DOI: 10.1038/s41598-024-83118-8](https://doi.org/10.1038/s41598-024-83118-8)

---

## 🌟 Key Innovations Over the Base Paper

1. **End-to-End Differentiable Band Selection:** Replaces the decoupled, unsupervised Pearson correlation threshold heuristic ($\tau = 0.6$) with a task-aware Gumbel-Softmax band selector optimized directly via gradient descent.
2. **Anti-Collinearity Repulsion Loss ($\mathcal{L}_{\text{repulse}}$):** Resolves the base paper's Salinas anomaly (where 7 adjacent contiguous bands $146\text{--}152$ were selected) by mathematically penalizing joint activation of spectrally proximate, correlated bands.
3. **Deep Learnable Bilateral Spatial Guided Filter:** Upgrades the static 1D recursive domain filter into an adaptive, texture-preserving spatial feature refiner.
4. **Residual Spectral-Spatial Attention Network (Res-SSAN):** Replaces the rudimentary 3-layer 3D-CNN with a high-capacity hybrid architecture combining volumetric convolutions, channel-wise Squeeze-and-Excitation attention, and spatial residual blocks.
5. **Leakage-Free Spatial Block Cross-Validation:** Demonstrates true geographic generalizability by separating train and test zones into disjoint spatial blocks, preventing patch-overlap data leakage.

---

## 📂 Project Structure

```
hyperspectralImageClassification/
├── data/
│   ├── dataset_manager.py          # Real AVIRIS loader + synthetic benchmark generator + spatial splitters
├── src/
│   ├── baseline/
│   │   ├── dual_partition.py       # Base Paper: Correlation + neighborhood weight + wavelet entropy
│   │   ├── recursive_filter.py     # Base Paper: Bi-hemispherical 1D recursive domain filter
│   │   └── base_3d_cnn.py          # Base Paper: Rudimentary 3-layer 3D-CNN
│   ├── models/
│   │   ├── diff_band_selector.py   # OURS: Differentiable Gumbel-Softmax band selector
│   │   ├── anti_collinearity.py    # OURS: Anti-Collinearity repulsion loss & MSD loss
│   │   ├── adaptive_guided_filter.py # OURS: Learnable bilateral spatial feature enhancer
│   │   └── spectral_spatial_backbone.py # OURS: Res-SSAN end-to-end classifier
│   └── utils/
│       ├── metrics.py              # OA, AA, Kappa, Mean Spectral Divergence (MSD), Entropy, Redundancy
│       └── visualization.py        # Publication-grade figures (band distribution, heatmaps, maps)
├── experiments/
│   └── run_comparison.py           # Automated comparative benchmark harness
├── results/                        # JSON metrics & benchmark evaluation summaries
├── figures/                        # Generated PNG plots & classification maps
├── run_all_benchmarks.py           # Master execution script
├── IMPROVEMENTS.md                 # In-depth technical comparison vs. base paper
├── RESEARCH_REPORT.md              # Formal IEEE-style research report & future work
└── requirements.txt                # Python package dependencies
```

---

## 🚀 Quick Start

### 1. Requirements
Ensure Python 3.9+ and PyTorch are installed:
```bash
pip install -r requirements.txt
```

### 2. Run Comprehensive Comparative Benchmark
To run the automated benchmark comparing the Base Paper model against Our Enhanced Model:
```bash
# Quick verification run (fast training)
python run_all_benchmarks.py --quick

# Full convergence benchmark on Indian Pines (Spatial Block Split)
python run_all_benchmarks.py --dataset IndianPines --split spatial_block

# Test under conventional Random Pixel Split
python run_all_benchmarks.py --dataset IndianPines --split random
```

---

## 📊 Outputs & Deliverables
- **Metrics Summary:** Saved to `results/<dataset>_benchmark_summary.json`
- **Band Selection Plots:** Saved to `figures/<dataset>_band_selection.png`
- **Correlation Heatmaps:** Saved to `figures/<dataset>_correlation_heatmaps.png`
- **Classification Maps:** Saved to `figures/<dataset>_classification_maps.png`
- **Technical Analysis:** See [IMPROVEMENTS.md](IMPROVEMENTS.md)
- **Full Research Paper:** See [RESEARCH_REPORT.md](RESEARCH_REPORT.md)
