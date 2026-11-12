"""
run_all_benchmarks.py
=====================
Master entry point to reproduce the base paper and evaluate our enhanced framework.

Usage:
    python run_all_benchmarks.py
    python run_all_benchmarks.py --quick
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run Hyperspectral Band Selection Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run with reduced epochs for fast verification")
    parser.add_argument("--dataset", type=str, default="IndianPines", choices=["IndianPines", "Salinas", "KSC"])
    parser.add_argument("--split", type=str, default="spatial_block", choices=["spatial_block", "random"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--fig_dir", type=str, default="figures")
    args = parser.parse_args()

    epochs = 4 if args.quick else 12

    cmd = [
        sys.executable,
        os.path.join("experiments", "run_comparison.py"),
        "--dataset", args.dataset,
        "--bands", "15",
        "--epochs", str(epochs),
        "--split_mode", args.split,
        "--output_dir", args.output_dir,
        "--fig_dir", args.fig_dir
    ]

    print(f"[*] Executing benchmark: {' '.join(cmd)}")
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
