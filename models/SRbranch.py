"""
Super-Resolution Branch for JEGAN-UNet3+.

Components:
    - TextureDecoder:     Reconstructs HR image from shared encoder features
    - PatchGANDiscriminator: 70x70 PatchGAN discriminator for adversarial training
    - SRBranch:           Combined generator wrapper (encoder + decoder + discriminator)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    """Hard sigmoid activation: ReLU6(x+3)/6."""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard swish activation."""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention module.
    Enhances directional feature representation along X/Y axes,
    particularly useful for photovoltaic array patterns.
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class CALayer(nn.Module):
    """Channel attention layer: adaptively selects useful channels for SR."""
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block for SR reconstruction."""
    def __init__(self, channels, reduction=16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            CALayer(channels, reduction)
        )

    def forward(self, x):
        return x + self.body(x)


class TextureDecoder(nn.Module):
    """
    Texture decoder for super-resolution reconstruction.

    Takes shallow features (h1, h2, h3) from the shared encoder and
    reconstructs the high-resolution image with fine-grained texture.

    Args:
        scale:        Upsampling factor (default: 4)
        in_ch_h1:     Input channels from encoder layer 1 (default: 64)
        in_ch_h2:     Input channels from encoder layer 2 (default: 128)
        in_ch_h3:     Input channels from encoder layer 3 (default: 256)
        base_ch:      Base feature channels (default: 64)
        num_rcab:     Number of RCAB blocks (default: 8)
        out_channels: Number of output image channels (default: 3, matching RGB)
        use_coordatt: Whether to use Coordinate Attention (default: True)

    Returns:
        sr_img:  Super-resolved image [B, out_channels, H*scale, W*scale]
        sr_feat: Texture features for segmentation branch injection
    """
    def __init__(self, scale=1, in_ch_h1=64, in_ch_h2=128, in_ch_h3=256,
                 base_ch=64, num_rcab=8, out_channels=3, use_coordatt=True):
        super().__init__()
        self.scale = scale
        self.use_coordatt = use_coordatt

        # Feature alignment
        self.align_h2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch_h2, base_ch, 1)
        )
        self.align_h3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch_h3, base_ch, 1)
        )
        self.align_h1 = nn.Conv2d(in_ch_h1, base_ch, 1)

        # Fusion and coordinate attention
        self.fusion = nn.Conv2d(base_ch * 3, base_ch, 3, padding=1)
        self.coord_att = CoordAtt(base_ch, base_ch) if use_coordatt else nn.Identity()

        # Deep texture reconstruction with RCABs
        self.recon_trunk = nn.Sequential(*[RCAB(base_ch) for _ in range(num_rcab)])
        self.trunk_conv = nn.Conv2d(base_ch, base_ch, 3, padding=1)

        # ESPCN sub-pixel upsampling
        self.upsample = self._make_espcns(base_ch, scale)

        # Final image reconstruction head
        self.img_recon = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def _make_espcns(self, in_ch, scale):
        """Generate recursive pixel-shuffle layers for power-of-2 upscaling."""
        layers = []
        cur_ch = in_ch
        cur_scale = scale
        while cur_scale > 1:
            layers.append(nn.Conv2d(cur_ch, cur_ch * 4, 3, padding=1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.ReLU(inplace=True))
            cur_scale //= 2
        return nn.Sequential(*layers)

    def forward(self, h1, h2, h3):
        # 1. Scale alignment
        h1_aligned = self.align_h1(h1)
        h2_up = self.align_h2(h2)
        h3_up = self.align_h3(h3)

        # 2. Fusion with coordinate attention
        fused = self.fusion(torch.cat([h1_aligned, h2_up, h3_up], dim=1))
        fused = self.coord_att(fused)

        # 3. Deep texture reconstruction (with residual)
        recon = self.recon_trunk(fused)
        sr_feat = fused + self.trunk_conv(recon)

        # 4. Upsampling and image generation
        up_feat = self.upsample(sr_feat)
        sr_img = self.img_recon(up_feat)

        return sr_img, sr_feat


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.

    Receives HR/SR images and outputs a feature map that classifies
    each local patch as real or fake.
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class SRBranch(nn.Module):
    """
    Super-resolution branch combining encoder, texture decoder, and discriminator.
    Can be trained independently or jointly with the segmentation branch.
    """
    def __init__(self, shared_encoder, texture_decoder, discriminator):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.texture_decoder = texture_decoder
        self.discriminator = discriminator

    def forward(self, x):
        h1, h2, h3, _, _ = self.shared_encoder(x)
        sr_img, sr_feat = self.texture_decoder(h1, h2, h3)
        return sr_img, sr_feat

    def get_features(self, x):
        """Return encoder features (for perceptual loss)."""
        return self.shared_encoder(x)
