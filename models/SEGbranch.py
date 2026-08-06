"""
Segmentation Branch for JEGAN-UNet3+.

Implements the PVSegmentationHeadUNet3Plus decoder with:
    - Full-scale skip connections (UNet3+ style)
    - ASPP bottleneck for multi-scale context
    - Sobel edge guidance
    - SR feature injection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DFeatureExtract import DFeatureExtract


def resize_like(x, ref):
    """Resize x to match the spatial size of ref."""
    return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)


def get_group_norm(channels, max_groups=8):
    """Pick the largest group count that evenly divides channels."""
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Module):
    """Conv + GroupNorm + ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            get_group_norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """Two sequential Conv-GN-ReLU blocks."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_channels, out_channels, 3),
            ConvGNAct(out_channels, out_channels, 3)
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    """Simplified Atrous Spatial Pyramid Pooling."""
    def __init__(self, in_channels, out_channels, atrous_rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for rate in atrous_rates:
            if rate == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    get_group_norm(out_channels),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=rate, dilation=rate, bias=False),
                    get_group_norm(out_channels),
                    nn.ReLU(inplace=True)
                ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            get_group_norm(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1, bias=False),
            get_group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]
        outs = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        outs.append(gp)
        x = torch.cat(outs, dim=1)
        return self.project(x)


class FixedSobel(nn.Module):
    """Fixed Sobel edge extraction via depthwise convolution."""
    def __init__(self, channels):
        super().__init__()
        self.conv_x = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv_y = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        with torch.no_grad():
            self.conv_x.weight.copy_(sobel_x.repeat(channels, 1, 1, 1))
            self.conv_y.weight.copy_(sobel_y.repeat(channels, 1, 1, 1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)


class EdgeGuide(nn.Module):
    """Extract edge / high-frequency guidance from h1."""
    def __init__(self, in_channels, edge_channels=64):
        super().__init__()
        self.reduce = ConvGNAct(in_channels, edge_channels, kernel_size=1, padding=0)
        self.sobel = FixedSobel(edge_channels)
        self.refine = nn.Sequential(
            ConvGNAct(edge_channels, edge_channels, 3),
            ConvGNAct(edge_channels, edge_channels, 3)
        )

    def forward(self, h1):
        x = self.reduce(h1)
        x = self.sobel(x)
        x = self.refine(x)
        return x


class SRFeatureGuide(nn.Module):
    """Inject SR features into the segmentation branch."""
    def __init__(self, sr_channels=64, seg_channels=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(sr_channels, seg_channels, 1),
            nn.GroupNorm(8, seg_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, sr_feat, seg_feat):
        sr_proj = self.adapter(sr_feat)
        return seg_feat + sr_proj


class FullScaleFusionBlock(nn.Module):
    """
    UNet3+ style full-scale fusion block.
    Aggregates features from all encoder scales at a target resolution.
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                get_group_norm(out_channels),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            DoubleConv(out_channels * len(in_channels_list), out_channels),
            nn.Dropout2d(0.1)
        )

    def forward(self, feats, ref_feat):
        assert len(feats) == len(self.proj), "Number of input features does not match initialization"
        target_size = ref_feat.shape[2:]
        outs = []
        for x, p in zip(feats, self.proj):
            x = p(x)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        return self.fuse(x)


class PVSegmentationHeadUNet3Plus(nn.Module):
    """
    Full segmentation decoder compatible with DFeatureExtract (h1~h5).

    Input features:
        h1: [B, 64,  H,   W]
        h2: [B, 128, H/2, W/2]
        h3: [B, 256, H/4, W/4]
        h4: [B, 512, H/8, W/8]
        h5: [B, 512, H/16,W/16]

    Returns:
        main_logits: [B, num_classes, H, W]
        aux_logits:  dict of deep supervision outputs (d2, d3, d4)
    """
    def __init__(self, num_classes=1, decoder_channels=64,
                 h1_channels=64, h2_channels=128, h3_channels=256,
                 h4_channels=512, h5_channels=512,
                 use_aspp=True, use_sobel=True, use_sr_inject=True):
        super().__init__()
        self.use_aspp = use_aspp
        self.use_sobel = use_sobel
        self.use_sr_inject = use_sr_inject

        # Bottleneck
        if use_aspp:
            self.bottleneck = ASPP(h5_channels, decoder_channels)
            bottleneck_out_channels = decoder_channels
        else:
            self.bottleneck = nn.Sequential(
                DoubleConv(h5_channels, decoder_channels),
                nn.Dropout2d(0.1)
            )
            bottleneck_out_channels = decoder_channels

        # Fusion stages (d4 -> d3 -> d2 -> d1)
        self.fuse_d4 = FullScaleFusionBlock(
            [h1_channels, h2_channels, h3_channels, h4_channels, bottleneck_out_channels],
            decoder_channels
        )
        self.fuse_d3 = FullScaleFusionBlock(
            [h1_channels, h2_channels, h3_channels, h4_channels, bottleneck_out_channels, decoder_channels],
            decoder_channels
        )
        self.fuse_d2 = FullScaleFusionBlock(
            [h1_channels, h2_channels, h3_channels, h4_channels, bottleneck_out_channels,
             decoder_channels, decoder_channels],
            decoder_channels
        )
        self.fuse_d1 = FullScaleFusionBlock(
            [h1_channels, h2_channels, h3_channels, h4_channels, bottleneck_out_channels,
             decoder_channels, decoder_channels, decoder_channels],
            decoder_channels
        )

        # SR feature injection
        if self.use_sr_inject:
            self.sr_feature_guide = SRFeatureGuide(sr_channels=decoder_channels, seg_channels=decoder_channels)

        # Sobel edge guidance with dynamic fusion
        if self.use_sobel:
            self.edge_guide = EdgeGuide(h1_channels, edge_channels=decoder_channels)
            self.final_fuse = nn.Sequential(
                DoubleConv(decoder_channels * 2, decoder_channels),
                nn.Dropout2d(0.1)
            )
        else:
            self.final_fuse = nn.Sequential(
                DoubleConv(decoder_channels, decoder_channels),
                nn.Dropout2d(0.1)
            )

        # Output heads
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.aux_d4 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.aux_d3 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.aux_d2 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, h1, h2, h3, h4, h5, sr_feat):
        # Bottleneck
        h5b = self.bottleneck(h5)

        # Decoder stages
        d4 = self.fuse_d4([h1, h2, h3, h4, h5b], ref_feat=h4)
        d4_up = F.interpolate(d4, size=h3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.fuse_d3([h1, h2, h3, h4, h5b, d4_up], ref_feat=h3)

        d4_up2 = F.interpolate(d4, size=h2.shape[2:], mode='bilinear', align_corners=False)
        d3_up = F.interpolate(d3, size=h2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.fuse_d2([h1, h2, h3, h4, h5b, d4_up2, d3_up], ref_feat=h2)

        d4_up3 = F.interpolate(d4, size=h1.shape[2:], mode='bilinear', align_corners=False)
        d3_up2 = F.interpolate(d3, size=h1.shape[2:], mode='bilinear', align_corners=False)
        d2_up = F.interpolate(d2, size=h1.shape[2:], mode='bilinear', align_corners=False)
        sr_feat = F.interpolate(sr_feat, size=h1.shape[2:], mode='bilinear', align_corners=False)

        d1 = self.fuse_d1(
            [h1, h2, h3, h4, h5b, d4_up3, d3_up2, d2_up],
            ref_feat=h1
        )

        # SR feature injection (residual style)
        if self.use_sr_inject:
            d1 = self.sr_feature_guide(sr_feat, d1)

        # Sobel edge guidance
        if self.use_sobel:
            edge_feat = self.edge_guide(h1)
            if self.use_sr_inject:
                sr_edge = torch.mean(sr_feat, dim=1, keepdim=True)
                edge_feat = edge_feat + sr_edge
            out = torch.cat([d1, edge_feat], dim=1)
        else:
            out = d1

        out = self.final_fuse(out)
        main_logits = self.classifier(out)

        # Deep supervision outputs
        aux_logits = {
            "d4": self.aux_d4(d4),
            "d3": self.aux_d3(d3),
            "d2": self.aux_d2(d2)
        }

        # Upsample to full resolution
        if main_logits.shape[2:] != h1.shape[2:]:
            main_logits = F.interpolate(main_logits, size=h1.shape[2:], mode='bilinear', align_corners=False)
        for k in aux_logits:
            aux_logits[k] = F.interpolate(aux_logits[k], size=h1.shape[2:], mode='bilinear', align_corners=False)

        return main_logits, aux_logits


if __name__ == "__main__":
    encoder = DFeatureExtract(in_channels=3, base_filters=64)
    seg_head = PVSegmentationHeadUNet3Plus(num_classes=1, decoder_channels=64)
    x = torch.randn(2, 3, 256, 256)
    h1, h2, h3, h4, h5 = encoder(x)
    main_logits, aux_logits = seg_head(h1, h2, h3, h4, h5, sr_feat=torch.randn(2, 64, 256, 256))
    print("main_logits:", main_logits.shape)
    print("aux_d4:", aux_logits["d4"].shape)
    print("aux_d3:", aux_logits["d3"].shape)
    print("aux_d2:", aux_logits["d2"].shape)
