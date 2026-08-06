"""
JEGAN-UNet3+ Inference Script

Loads a trained model checkpoint and performs inference on test data.
Computes MSE, SSIM, and IoU metrics and saves visualizations.

Usage:
    python predict.py --checkpoint checkpoints/best.pth --input_dir dataset/test
"""
import torch
from pathlib import Path
import numpy as np
import tifffile
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from train import PVJointModel, Config


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "checkpoints/best.pth"

INPUT_DIR = Path("dataset/test/LR")
HR_DIR = Path("dataset/test/HR")
LABEL_DIR = Path("dataset/test/label")

OUTPUT_VIS_DIR = Path("results/vis")
OUTPUT_SR_DIR = Path("results/sr")
OUTPUT_SEG_DIR = Path("results/seg")


def to_channel_first(img):
    """Return image data as [C, H, W]."""
    if img.ndim == 2:
        return img[None, :, :]
    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if img.shape[0] <= 4 and img.shape[1] > 4 and img.shape[2] > 4:
        return img
    return np.transpose(img, (2, 0, 1))


class TestDataset(Dataset):
    """Dataset for inference on test images."""
    def __init__(self, lr_dir, hr_dir, label_dir, num_bands=Config.num_bands):
        self.lr_files = sorted(list(Path(lr_dir).glob("*.tif")))
        self.hr_files = sorted(list(Path(hr_dir).glob("*.tif")))
        self.label_files = sorted(list(Path(label_dir).glob("*.png")))
        self.num_bands = num_bands
        assert len(self.lr_files) == len(self.hr_files) == len(self.label_files), \
            "Number of files does not match"

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr = to_channel_first(tifffile.imread(self.lr_files[idx]).astype(np.float32))
        hr = to_channel_first(tifffile.imread(self.hr_files[idx]).astype(np.float32))
        label = imageio.imread(self.label_files[idx]).astype(np.float32)
        if lr.shape[0] < self.num_bands or hr.shape[0] < self.num_bands:
            raise ValueError(f"Expected at least {self.num_bands} bands in LR/HR images")
        lr = lr[:self.num_bands, :, :]
        hr = hr[:self.num_bands, :, :]
        if label.ndim == 2:
            label = label[None, :, :]
        elif label.ndim == 3:
            label = np.transpose(label, (2, 0, 1))
        lr = torch.from_numpy(lr / 255.0).float()
        hr = torch.from_numpy(hr / 255.0).float()
        label = torch.from_numpy(label / 255.0).float()
        label[label > 0] = 1.0
        return lr, hr, label, self.lr_files[idx].name


def load_joint_model(checkpoint_path, device, opt):
    """Load the trained PVJointModel from a checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = PVJointModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def mse(a, b):
    return ((a - b) ** 2).mean()


def ssim_metric(a, b):
    a_np = np.clip(a * 255, 0, 255).astype(np.uint8)
    b_np = np.clip(b * 255, 0, 255).astype(np.uint8)
    s = 0
    for c in range(a_np.shape[0]):
        s += compare_ssim(a_np[c], b_np[c], data_range=255)
    return s / a_np.shape[0]


def iou_metric(pred, label):
    pred_bin = (pred > 0.5).astype(np.uint8).flatten()
    label_bin = (label > 0.5).astype(np.uint8).flatten()
    return jaccard_score(label_bin, pred_bin, zero_division=0)


def to_rgb(img):
    """Convert an image tensor to RGB for display."""
    if img.shape[0] >= 3:
        return np.transpose(img[:3], (1, 2, 0))
    else:
        return np.transpose(np.tile(img, (3, 1, 1)), (1, 2, 0))


def inference_and_visualize():
    """Run inference and save results + visualizations."""
    OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SR_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SEG_DIR.mkdir(parents=True, exist_ok=True)

    opt = Config()
    model = load_joint_model(CHECKPOINT_PATH, DEVICE, opt)

    dataset = TestDataset(INPUT_DIR, HR_DIR, LABEL_DIR, num_bands=opt.num_bands)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for lr, hr, label, fname in tqdm(loader, total=len(loader), desc="Inference", ncols=100):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            label = label.to(DEVICE)

            sr, seg_probs = model(lr)

            lr_np = lr.squeeze(0).cpu().numpy()
            hr_np = hr.squeeze(0).cpu().numpy()
            sr_np = sr.squeeze(0).cpu().numpy()
            seg_np = seg_probs[0, 0].cpu().numpy()
            seg_bin = (seg_np > 0.5).astype(np.float32)
            label_np = label[0, 0].cpu().numpy()

            mse_val = mse(sr_np, hr_np)
            ssim_val = ssim_metric(sr_np, hr_np)
            iou_val = iou_metric(seg_bin, label_np)

            tqdm.write(f"{fname[0]} | MSE:{mse_val:.4f} | SSIM:{ssim_val:.4f} | IoU:{iou_val:.4f}")

            # Save SR image
            sr_save = np.clip(sr_np * 255, 0, 255).astype(np.uint8)
            if sr_save.ndim == 3:
                sr_save = np.transpose(sr_save, (1, 2, 0))
            tifffile.imwrite(OUTPUT_SR_DIR / fname[0], sr_save)

            # Save segmentation mask
            imageio.imwrite(
                OUTPUT_SEG_DIR / fname[0].replace('.tif', '.png'),
                (seg_bin * 255).astype(np.uint8)
            )

            # Visualize
            H, W = sr_np.shape[1:]
            gap = 5
            imgs = [
                to_rgb(lr_np),
                to_rgb(sr_np),
                to_rgb(hr_np),
                np.stack([seg_bin]*3, -1),
                np.stack([label_np]*3, -1)
            ]
            canvas = np.ones((H, 5*W + 4*gap, 3), dtype=np.float32)
            for i, img in enumerate(imgs):
                img = np.clip(img, 0, 1)
                canvas[:, i*(W+gap):i*(W+gap)+W] = img

            plt.figure(figsize=(15, 5))
            plt.imshow(canvas)
            plt.axis('off')
            plt.title(f"MSE:{mse_val:.4f} | SSIM:{ssim_val:.4f} | IoU:{iou_val:.4f}")
            plt.savefig(OUTPUT_VIS_DIR / fname[0].replace('.tif', '.png'), bbox_inches='tight', pad_inches=0.1)
            plt.close()


if __name__ == "__main__":
    inference_and_visualize()
