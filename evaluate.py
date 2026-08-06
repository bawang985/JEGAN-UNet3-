"""
JEGAN-UNet3+ Evaluation Script

Computes global metrics (IoU, Precision, Recall, F1) by comparing
predicted segmentation masks against ground truth labels.
Supports evaluation of multiple model variants for ablation studies.

Usage:
    python evaluate.py --pred_dir results/seg --gt_dir dataset/test/label
"""
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXT = (".png", ".tif", ".tiff")


def load_binary_mask(path):
    """Load an image as a binary mask (0 or 1)."""
    img = Image.open(path).convert("L")
    mask = np.array(img)
    mask = (mask > 0).astype(np.uint8)
    return mask


def compute_counts(pred, gt):
    """Return pixel-level TP, FP, FN counts."""
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    return tp, fp, fn


def build_file_map(folder):
    """Build filename (no ext) -> full path mapping."""
    file_map = {}
    for f in os.listdir(folder):
        if f.lower().endswith(SUPPORTED_EXT):
            name = os.path.splitext(f)[0]
            file_map[name] = os.path.join(folder, f)
    return file_map


def evaluate(pred_dir, gt_dir):
    """Compute global evaluation metrics."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    pred_map = build_file_map(pred_dir)
    gt_map = build_file_map(gt_dir)

    if len(pred_map) == 0:
        print("Prediction folder is empty!")
        return

    skipped = 0
    for name, pred_path in tqdm(pred_map.items(), desc="Evaluating", unit="img"):
        if name not in gt_map:
            skipped += 1
            continue
        gt_path = gt_map[name]
        pred = load_binary_mask(pred_path)
        gt = load_binary_mask(gt_path)
        if pred.shape != gt.shape:
            print(f"Size mismatch, skipping: {name}")
            continue
        tp, fp, fn = compute_counts(pred, gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    if skipped > 0:
        print(f"Skipped {skipped} images without matching ground truth")

    if (total_tp + total_fp + total_fn) == 0:
        print("No valid foreground pixels found!")
        return

    global_iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    global_precision = total_tp / (total_tp + total_fp + 1e-8)
    global_recall = total_tp / (total_tp + total_fn + 1e-8)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-8)

    print("\n====== Global Evaluation Metrics ======")
    print(f"Global IoU       : {global_iou:.4f}")
    print(f"Global Precision : {global_precision:.4f}")
    print(f"Global Recall    : {global_recall:.4f}")
    print(f"Global F1-score  : {global_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation results")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to predicted masks")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to ground truth masks")
    args = parser.parse_args()
    evaluate(args.pred_dir, args.gt_dir)
