"""
End-to-end demo script for JEGAN-UNet3+.

Generates synthetic data, trains for a few epochs, and runs inference.

Usage:
    python demo/run_demo.py
"""
import os
import sys
import subprocess
from pathlib import Path


def main():
    demo_dir = Path(__file__).parent.parent / "demo_data"
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

    print("=" * 60)
    print("JEGAN-UNet3+ Demo")
    print("=" * 60)

    # Step 1: Generate synthetic data
    print("\n[1/3] Generating synthetic data...")
    sys.stdout.flush()
    ret = subprocess.call([
        sys.executable, "demo/generate_synthetic_data.py",
        "--output_dir", str(demo_dir),
        "--num_samples", "50"
    ])
    if ret != 0:
        print("Data generation failed.")
        return

    # Step 2: Quick training (just 3 epochs for demo purposes)
    print("\n[2/3] Running quick training (3 epochs for demo)...")
    sys.stdout.flush()
    ret = subprocess.call([
        sys.executable, "train.py",
    ])
    if ret != 0:
        print("Training failed (expected if no CUDA / full dataset).")
        print("Skipping inference step...")
        return

    # Step 3: Run inference
    print("\n[3/3] Running inference on test set...")
    sys.stdout.flush()
    ret = subprocess.call([
        sys.executable, "predict.py",
    ])

    print("\n" + "=" * 60)
    print("Demo completed!")
    print(f"Results saved to: {demo_dir.parent / 'results'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
