"""
JEGAN-UNet3+ Joint Training Script

Jointly trains the super-resolution (SR) branch and the semantic segmentation
branch in a unified GAN framework for photovoltaic panel detection from
remote sensing imagery.

Usage:
    python train.py                             # train with default config
    python train.py --resume --data_dir ./data  # resume from checkpoint
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

import rasterio
import shutil
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer
from math import exp

from models.DFeatureExtract import DFeatureExtract
from models.SRbranch import TextureDecoder, PatchGANDiscriminator
from models.SEGbranch import PVSegmentationHeadUNet3Plus
from utils.utils import get_logger, AverageMeter, save_checkpoint, log_csv
from configs.default_config import Config


# ============================================================================
# Dataset
# ============================================================================
class PatchDataset(Dataset):
    """Read LR, HR patches and corresponding labels from TIF/PNG files."""
    def __init__(self, lr_folder, hr_folder, label_folder, num_bands=Config.num_bands):
        self.lr_files = sorted(list(Path(lr_folder).rglob('*.tif')))
        self.hr_files = sorted(list(Path(hr_folder).rglob('*.tif')))
        self.label_files = sorted(list(Path(label_folder).rglob('*.png')))
        self.num_bands = num_bands
        assert len(self.lr_files) == len(self.hr_files) == len(self.label_files), \
            "LR, HR, and label patch counts do not match"

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        with rasterio.open(self.lr_files[idx]) as src:
            lr = src.read().astype('float32') / 255.0
        with rasterio.open(self.hr_files[idx]) as src:
            hr = src.read().astype('float32') / 255.0
        with rasterio.open(self.label_files[idx]) as src:
            label = src.read(1).astype('float32')
        if lr.shape[0] < self.num_bands or hr.shape[0] < self.num_bands:
            raise ValueError(f"Expected at least {self.num_bands} bands in LR/HR images")
        lr = lr[:self.num_bands, :, :]
        hr = hr[:self.num_bands, :, :]
        label[label > 0] = 1.0
        label = label[None, :, :]
        return torch.tensor(lr), torch.tensor(hr), torch.tensor(label)


# ============================================================================
# Loss Functions for SR Branch
# ============================================================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


class MSSSIMLoss(nn.Module):
    """Multi-scale SSIM as a loss (1 - MSSSIM)."""
    def __init__(self, normalize=True):
        super(MSSSIMLoss, self).__init__()
        self.normalize = normalize

    def forward(self, x, y):
        return 1.0 - msssim(x.float(), y.float(), normalize=self.normalize)


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for imbalanced segmentation.
    alpha: weight for false negatives (missed detections), default 0.7
    beta:  weight for false positives, default 0.3
    gamma: focusing parameter, default 0.75
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        target = target.view(-1)
        TP = (probs * target).sum()
        FP = ((1 - target) * probs).sum()
        FN = (target * (1 - probs)).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        focal_tversky = torch.clamp(1 - tversky, min=1e-6) ** self.gamma
        return focal_tversky


class PVJointSegLoss(nn.Module):
    """Combined BCE + Focal Tversky loss for segmentation."""
    def __init__(self, weight_bce=0.5, weight_ft=1.0):
        super(PVJointSegLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ft = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
        self.weight_bce = weight_bce
        self.weight_ft = weight_ft

    def forward(self, logits, target):
        loss_bce = self.bce(logits, target)
        loss_ft = self.ft(logits, target)
        return self.weight_bce * loss_bce + self.weight_ft * loss_ft


class CharbonnierLoss(nn.Module):
    """Smooth L1 loss robust to extreme pixel values."""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 early layer features."""
    def __init__(self, feature_layer=2):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.loss_network = nn.Sequential(*list(vgg.children())[:feature_layer+1]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        sr = sr[:, :3, :, :]
        hr = hr[:, :3, :, :]
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        sr_feat = self.loss_network(sr)
        hr_feat = self.loss_network(hr)
        return F.mse_loss(sr_feat, hr_feat)


class GradientLoss(nn.Module):
    """Gradient difference loss for enforcing sharp edges."""
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, gen_img, gt_img):
        gen_rgb = gen_img[:, :3, :, :]
        gt_rgb = gt_img[:, :3, :, :]
        grad_gen_x = torch.abs(gen_rgb[:, :, :, 1:] - gen_rgb[:, :, :, :-1])
        grad_gen_y = torch.abs(gen_rgb[:, :, 1:, :] - gen_rgb[:, :, :-1, :])
        grad_gt_x = torch.abs(gt_rgb[:, :, :, 1:] - gt_rgb[:, :, :, :-1])
        grad_gt_y = torch.abs(gt_rgb[:, :, 1:, :] - gt_rgb[:, :, :-1, :])
        loss = torch.mean(torch.abs(grad_gen_x - grad_gt_x)) + \
               torch.mean(torch.abs(grad_gen_y - grad_gt_y))
        return loss


class CosineSimilarityLoss(nn.Module):
    """Cosine similarity loss for color consistency under overexposure."""
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, gen_img, gt_img):
        gen_rgb = gen_img[:, :3, :, :].float()
        gt_rgb = gt_img[:, :3, :, :].float()
        cos_sim = F.cosine_similarity(gen_rgb, gt_rgb, dim=1, eps=1e-8)
        return 1.0 - torch.mean(cos_sim)


# ============================================================================
# Joint Model: Encoder + SR Decoder + Seg Decoder + Discriminator
# ============================================================================
class PVJointModel(nn.Module):
    """Container for all network components."""
    def __init__(self, use_cbam=True, use_coordatt=True, use_sobel=True, use_sr_inject=True):
        super(PVJointModel, self).__init__()
        opt = Config()
        self.shared_encoder = DFeatureExtract(in_channels=opt.num_bands, base_filters=64, use_cbam=use_cbam)
        self.texture_decoder = TextureDecoder(
            scale=1, in_ch_h1=64, in_ch_h2=128, in_ch_h3=256,
            out_channels=opt.num_bands, use_coordatt=use_coordatt
        )
        self.seg_head = PVSegmentationHeadUNet3Plus(
            num_classes=1, decoder_channels=64,
            use_sobel=use_sobel, use_sr_inject=use_sr_inject
        )
        self.discriminator = PatchGANDiscriminator(in_channels=opt.num_bands)

    def forward(self, lr_img):
        h1, h2, h3, h4, h5 = self.shared_encoder(lr_img)
        sr_img, sr_feat = self.texture_decoder(h1, h2, h3)
        main_seg_logits, aux_logits = self.seg_head(h1, h2, h3, h4, h5, sr_feat)
        seg_probs = torch.sigmoid(main_seg_logits)
        return sr_img, seg_probs


# ============================================================================
# Experiment: Training Workflow
# ============================================================================
class Experiment:
    def __init__(self, option):
        self.device = torch.device('cuda' if torch.cuda.is_available() and option.cuda else 'cpu')
        self.opt = option
        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.best = self.train_dir / 'best.pth'
        self.last_model = self.train_dir / 'joint_model.pth'

        self.logger = get_logger()
        self.logger.info('Initializing models...')

        self.model = PVJointModel(
            use_cbam=self.opt.use_cbam,
            use_coordatt=self.opt.use_coordatt,
            use_sobel=self.opt.use_sobel,
            use_sr_inject=self.opt.use_sr_inject
        ).to(self.device)

        # Losses
        self.criterion_char = CharbonnierLoss().to(self.device)
        self.criterion_percep = VGGPerceptualLoss().to(self.device)
        self.criterion_msssim = MSSSIMLoss().to(self.device)
        self.criterion_seg = PVJointSegLoss().to(self.device)
        self.criterion_grad = GradientLoss().to(self.device)
        self.criterion_cos = CosineSimilarityLoss().to(self.device)

        # Optimizers
        params_G = list(self.model.shared_encoder.parameters()) + \
                   list(self.model.texture_decoder.parameters()) + \
                   list(self.model.seg_head.parameters())
        self.g_optimizer = optim.AdamW(params_G, lr=option.lr, weight_decay=1e-4)
        self.d_optimizer = optim.AdamW(self.model.discriminator.parameters(), lr=option.lr, betas=(0.9, 0.999))

        # Mixed precision
        self.scaler = GradScaler()
        self.start_epoch = 0

        # Resume if requested
        if self.opt.resume:
            self.load_checkpoint()

    def load_checkpoint(self):
        if self.last_model.exists():
            self.logger.info(f"Loading checkpoint '{self.last_model}'")
            checkpoint = torch.load(self.last_model, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            if 'g_optimizer' in checkpoint:
                self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            if 'd_optimizer' in checkpoint:
                self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            if 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Resuming training from epoch {self.start_epoch}")
        else:
            self.logger.info(f"No checkpoint found, starting from scratch.")

    def train_on_epoch(self, epoch, data_loader):
        """
        Train one epoch with curriculum learning.
        Segmentation loss is gradually introduced after epoch 10.
        """
        self.model.train()
        epg_loss = AverageMeter()
        eppd_loss = AverageMeter()
        epseg_loss = AverageMeter()
        epg_error = AverageMeter()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}", ncols=100)

        lambda_seg = min(1.0, max(0.0, (epoch - 10) / 40.0))

        for idx, data in pbar:
            lr, hr, label = [d.to(self.device) for d in data]

            # --- Update Discriminator ---
            for p in self.model.discriminator.parameters():
                p.requires_grad = True
            self.d_optimizer.zero_grad()
            with autocast():
                h1, h2, h3, h4, h5 = self.model.shared_encoder(lr)
                sr_img, _ = self.model.texture_decoder(h1, h2, h3)
                real_pred = self.model.discriminator(hr)
                fake_pred = self.model.discriminator(sr_img.detach())
                loss_D_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
                loss_D_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
                d_loss = (loss_D_real + loss_D_fake) * 0.5
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.d_optimizer)

            # --- Update Generator + Segmenter ---
            for p in self.model.discriminator.parameters():
                p.requires_grad = False
            self.g_optimizer.zero_grad()
            with autocast():
                h1, h2, h3, h4, h5 = self.model.shared_encoder(lr)
                sr_img, sr_feat = self.model.texture_decoder(h1, h2, h3)
                main_logits, aux_logits = self.model.seg_head(h1, h2, h3, h4, h5, sr_feat)

                # SR losses
                fake_pred_for_G = self.model.discriminator(sr_img)
                loss_adv = F.binary_cross_entropy_with_logits(fake_pred_for_G, torch.ones_like(fake_pred_for_G))
                loss_char = self.criterion_char(sr_img, hr)
                loss_perc = self.criterion_percep(sr_img, hr)
                loss_msssim = self.criterion_msssim(sr_img, hr)
                loss_grad = self.criterion_grad(sr_img, hr)
                loss_cos = self.criterion_cos(sr_img, hr)
                loss_sr_total = (
                    1.0 * loss_char +
                    0.5 * loss_msssim +
                    0.3 * loss_grad +
                    0.2 * loss_cos +
                    0.1 * loss_perc +
                    0.005 * loss_adv
                )

                # Segmentation loss
                if lambda_seg > 0:
                    loss_seg_main = self.criterion_seg(main_logits, label)
                    loss_seg_aux = (
                        self.criterion_seg(aux_logits['d4'], label) * 0.2 +
                        self.criterion_seg(aux_logits['d3'], label) * 0.3 +
                        self.criterion_seg(aux_logits['d2'], label) * 0.4
                    )
                    seg_loss = loss_seg_main + loss_seg_aux
                else:
                    seg_loss = torch.tensor(0.0, device=self.device)

                g_total = loss_sr_total + lambda_seg * seg_loss

            self.scaler.scale(g_total).backward()
            self.scaler.unscale_(self.g_optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.g_optimizer)
            self.scaler.update()

            # Logging
            mse = F.mse_loss(sr_img.detach(), hr).item()
            epg_loss.update(g_total.item())
            eppd_loss.update(d_loss.item())
            epseg_loss.update(seg_loss.item() if lambda_seg > 0 else 0.0)
            epg_error.update(mse)
            pbar.set_postfix(G=g_total.item(), D=d_loss.item(),
                             Seg=seg_loss.item() if lambda_seg > 0 else 0.0, MSE=mse)

        self.logger.info(
            f"Epoch[{epoch}] - G:{epg_loss.avg:.4f} D:{eppd_loss.avg:.4f} "
            f"Seg:{epseg_loss.avg:.4f} MSE:{epg_error.avg:.4f}"
        )
        return epg_loss.avg, eppd_loss.avg, epseg_loss.avg, epg_error.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.model.eval()
        epoch_error = AverageMeter()
        for lr, hr, _ in data_loader:
            lr, hr = lr.to(self.device), hr.to(self.device)
            sr_img, _ = self.model(lr)
            error = F.mse_loss(sr_img, hr)
            epoch_error.update(error.item())
        return epoch_error.avg

    def train(self, train_loader, val_loader, epochs=200):
        least_error = float('inf')
        for epoch in range(self.start_epoch, epochs):
            g_loss, d_loss, seg_loss, g_error = self.train_on_epoch(epoch, train_loader)
            val_error = self.test_on_epoch(val_loader)
            log_csv(self.history, [epoch, g_loss, d_loss, seg_loss, g_error, val_error],
                    header=['epoch', 'g_loss', 'd_loss', 'seg_loss', 'mse', 'val_error'])

            save_checkpoint(self.model, self.g_optimizer, self.d_optimizer, self.scaler, epoch, self.last_model)

            if val_error < least_error:
                least_error = val_error
                shutil.copy(str(self.last_model), str(self.best))

            if (epoch + 1) % 5 == 0:
                hist_model = self.train_dir / f'joint_model_epoch{epoch + 1}.pth'
                save_checkpoint(self.model, self.g_optimizer, self.d_optimizer, self.scaler, epoch, hist_model)
                print(f"Saved checkpoint at epoch {epoch + 1}")

            print(f"Epoch {epoch}: G={g_loss:.4f} D={d_loss:.4f} Seg={seg_loss:.4f} Val={val_error:.6f}")


if __name__ == '__main__':
    opt = Config()

    torch.manual_seed(2020)
    if opt.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(2020)

    train_dataset = PatchDataset(
        opt.data_dir / 'train' / 'LR',
        opt.data_dir / 'train' / 'HR',
        opt.data_dir / 'train' / 'label',
        num_bands=opt.num_bands
    )
    val_dataset = PatchDataset(
        opt.data_dir / 'val' / 'LR',
        opt.data_dir / 'val' / 'HR',
        opt.data_dir / 'val' / 'label',
        num_bands=opt.num_bands
    )

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    experiment = Experiment(opt)
    experiment.train(train_loader, val_loader, epochs=opt.epochs)
