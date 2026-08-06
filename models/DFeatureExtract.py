import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module (part of CBAM)."""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial attention module (part of CBAM)."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(concat)
        return self.sigmoid(att) * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial attention)."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DoubleConv(nn.Module):
    """Two sequential convolutions with BN + ReLU."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DFeatureExtract(nn.Module):
    """
    Shared feature extractor based on the UNet3+ encoder structure.

    Outputs five multi-scale feature maps (h1~h5) with optional CBAM
    attention augmentation.

    Output description:
        h1: 1/1 resolution, 64 channels  (fine detail, edges)
        h2: 1/2 resolution, 128 channels
        h3: 1/4 resolution, 256 channels
        h4: 1/8 resolution, 512 channels
        h5: 1/16 resolution, 512 channels (bottleneck, semantic context)

    Usage:
        encoder = DFeatureExtract(in_channels=3, base_filters=64)
        h1, h2, h3, h4, h5 = encoder(x)
    """
    def __init__(self, in_channels=3, base_filters=64, use_cbam=True):
        super(DFeatureExtract, self).__init__()
        self.use_cbam = use_cbam

        # Encoder layer 1 (no downsampling)
        self.conv1 = DoubleConv(in_channels, base_filters)
        self.cbam1 = CBAM(base_filters) if use_cbam else nn.Identity()

        # Encoder layer 2 (2x downsampling)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_filters, base_filters * 2)
        self.cbam2 = CBAM(base_filters * 2) if use_cbam else nn.Identity()

        # Encoder layer 3 (4x downsampling)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_filters * 2, base_filters * 4)
        self.cbam3 = CBAM(base_filters * 4) if use_cbam else nn.Identity()

        # Encoder layer 4 (8x downsampling)
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(base_filters * 4, base_filters * 8)
        self.cbam4 = CBAM(base_filters * 8) if use_cbam else nn.Identity()

        # Encoder layer 5 (16x downsampling, bottleneck)
        self.maxpool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(base_filters * 8, base_filters * 8)
        # Bottleneck: no CBAM here; ASPP or dilated conv can be added externally

    def forward(self, x):
        h1 = self.cbam1(self.conv1(x))

        x = self.maxpool1(h1)
        h2 = self.cbam2(self.conv2(x))

        x = self.maxpool2(h2)
        h3 = self.cbam3(self.conv3(x))

        x = self.maxpool3(h3)
        h4 = self.cbam4(self.conv4(x))

        x = self.maxpool4(h4)
        h5 = self.conv5(x)

        return h1, h2, h3, h4, h5


if __name__ == '__main__':
    encoder = DFeatureExtract(in_channels=3, base_filters=64)
    dummy = torch.randn(2, 3, 256, 256)
    h1, h2, h3, h4, h5 = encoder(dummy)
    print("h1 shape:", h1.shape)
    print("h2 shape:", h2.shape)
    print("h3 shape:", h3.shape)
    print("h4 shape:", h4.shape)
    print("h5 shape:", h5.shape)
