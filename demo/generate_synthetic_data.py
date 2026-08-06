"""
Synthetic Data Generator for JEGAN-UNet3+

Since the original remote sensing dataset cannot be shared publicly,
this script generates synthetic patches (LR, HR, and label masks) for
demonstration and testing purposes.

Usage:
    python demo/generate_synthetic_data.py --output_dir ./demo_data --num_samples 100
"""
import os
import argparse
import numpy as np
from PIL import Image
import tifffile
from tqdm import tqdm


def generate_synthetic_patch(size=512, num_bands=3):
    """
    Generate a single synthetic patch with simple geometric shapes
    representing photovoltaic panels.

    Returns:
        lr:   Degraded low-resolution-source image on the same grid (num_bands, size, size)
        hr:   High-resolution image (num_bands, size, size)
        label: Binary segmentation mask (size, size)
    """
    # Generate HR image with random backgrounds and panel-like rectangles
    hr = np.random.uniform(0.3, 0.8, (num_bands, size, size)).astype(np.float32)

    # Generate binary label (random rectangular panels)
    label = np.zeros((size, size), dtype=np.float32)
    num_panels = np.random.randint(3, 12)
    for _ in range(num_panels):
        pw, ph = np.random.randint(20, 80), np.random.randint(20, 80)
        px, py = np.random.randint(0, size - pw), np.random.randint(0, size - ph)
        # Fill panel with brighter values and distinctive spectral signature
        panel_rgb = np.random.uniform(0.5, 1.0, (num_bands, ph, pw))
        hr[:, py:py+ph, px:px+pw] = panel_rgb
        label[py:py+ph, px:px+pw] = 1.0

    # Add some Gaussian noise
    hr = np.clip(hr + np.random.normal(0, 0.02, hr.shape), 0, 1)

    # Generate LR-like input on the same grid by downsampling and resizing back.
    lr = np.empty_like(hr)
    for band_idx in range(num_bands):
        band = Image.fromarray((hr[band_idx] * 255).astype(np.uint8))
        band = band.resize((size // 4, size // 4), Image.BICUBIC)
        band = band.resize((size, size), Image.BICUBIC)
        lr[band_idx] = np.asarray(band).astype(np.float32) / 255.0
    lr = np.clip(lr + np.random.normal(0, 0.02, lr.shape), 0, 1)

    return lr, hr, label


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for JEGAN-UNet3+")
    parser.add_argument("--output_dir", type=str, default="./demo_data",
                        help="Output directory for synthetic data")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--size", type=int, default=512,
                        help="HR image size (LR will be size//4)")
    args = parser.parse_args()

    splits = {
        'train': int(args.num_samples * 0.7),
        'val': int(args.num_samples * 0.15),
        'test': int(args.num_samples * 0.15),
    }

    for split, n in splits.items():
        lr_dir = os.path.join(args.output_dir, split, 'LR')
        hr_dir = os.path.join(args.output_dir, split, 'HR')
        label_dir = os.path.join(args.output_dir, split, 'label')
        os.makedirs(lr_dir, exist_ok=True)
        os.makedirs(hr_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for i in tqdm(range(n), desc=f"Generating {split}"):
            lr, hr, label = generate_synthetic_patch(size=args.size)

            # Save LR as TIF
            lr_path = os.path.join(lr_dir, f"patch_{i:04d}.tif")
            tifffile.imwrite(lr_path, (np.transpose(lr * 255, (1, 2, 0)).astype(np.uint8)))

            # Save HR as TIF
            hr_path = os.path.join(hr_dir, f"patch_{i:04d}.tif")
            tifffile.imwrite(hr_path, (np.transpose(hr * 255, (1, 2, 0)).astype(np.uint8)))

            # Save label as PNG
            label_path = os.path.join(label_dir, f"patch_{i:04d}.png")
            Image.fromarray((label * 255).astype(np.uint8)).save(label_path)

    print(f"\nSynthetic data generated at: {os.path.abspath(args.output_dir)}")
    print(f"  Train: {splits['train']} samples")
    print(f"  Val:   {splits['val']} samples")
    print(f"  Test:  {splits['test']} samples")


if __name__ == "__main__":
    main()
