import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block to re-weight spectral channels."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x shape: (B, C, L)
        w = x.mean(dim=2)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(2)


class SEResidualBottleneck1D(nn.Module):
    """Deep Bottleneck ResNet Block with Dilated Convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dropout=0.1, reduction=4):
        super().__init__()

        # The "Bottleneck" shrinks the channels by a factor of 4
        mid_channels = out_channels // reduction

        # Padding formula for dilated convolutions to keep array shapes aligned
        padding = (kernel_size - 1) * dilation // 2

        self.conv_path = nn.Sequential(
            # 1. Shrink (1x1 Convolution)
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),

            # 2. Spatial Convolution (The Dilated one)
            nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Dropout(dropout),

            # 3. Expand (1x1 Convolution)
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        # Skip connection needs to handle channel changes or stride downsampling
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.se = SEBlock(out_channels)

    def forward(self, x):
        out = self.conv_path(x)
        out = out + self.skip(x)
        return F.gelu(self.se(out))  # Using GELU here too

class MultiScaleStem(nn.Module):
    """Initial convolutional stem that captures multi-scale features from the raw spectrum."""
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        # Parallel convolutions looking at narrow, medium, and wide features
        self.conv_narrow = nn.Conv1d(in_channels, out_channels//3, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv_medium = nn.Conv1d(in_channels, out_channels//3, kernel_size=15, stride=2, padding=7, bias=False)
        self.conv_wide   = nn.Conv1d(in_channels, out_channels - 2*(out_channels//3), kernel_size=31, stride=2, padding=15, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_narrow(x)
        x2 = self.conv_medium(x)
        x3 = self.conv_wide(x)
        x = torch.cat([x1, x2, x3], dim=1) # Combine all three views
        return self.pool(F.gelu(self.bn(x)))


class SpectraClassifier(nn.Module):
    """Refined 1D SE-ResNet for SDSS classification."""

    def __init__(self, num_classes=3, aux_features=6, dropout=0.3):
        super().__init__()

        # Initial feature extraction (Stem)
        self.stem = MultiScaleStem(in_channels=1, out_channels=32)

        # SE-Residual Blocks

        # Stage 1 (32 -> 64) - 2 Blocks
        self.block1a = SEResidualBottleneck1D(32, 64, stride=2)
        self.block1b = SEResidualBottleneck1D(64, 64, dilation=2)

        # Stage 2 (64 -> 128) - 3 Blocks
        self.block2a = SEResidualBottleneck1D(64, 128, stride=2)
        self.block2b = SEResidualBottleneck1D(128, 128, dilation=2)
        self.block2c = SEResidualBottleneck1D(128, 128, dilation=4)

        # Stage 3 (128 -> 256) - 3 Blocks
        self.block3a = SEResidualBottleneck1D(128, 256, stride=2)
        self.block3b = SEResidualBottleneck1D(256, 256, dilation=2)
        self.block3c = SEResidualBottleneck1D(256, 256, dilation=4)

        # Stage 4 (256 -> 512) - 3 Blocks
        self.block4a = SEResidualBottleneck1D(256, 512, stride=2)
        self.block4b = SEResidualBottleneck1D(512, 512, dilation=2)
        self.block4c = SEResidualBottleneck1D(512, 512, dilation=4)

        # Stage 5 (512 -> 1024) - 1 Block (Keep it brief before Attention)
        self.block5a = SEResidualBottleneck1D(512, 1024, stride=2)

        # Powerboost: Add a multi-head attention layer after the convolutional blocks to capture long-range dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=1024,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head (handles flux features + optional scalar data)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + aux_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, flux, aux=None):
        # Handle input shape from new DataLoader: [Batch, 1, Length]
        if flux.dim() == 2:
            flux = flux.unsqueeze(1)

        x = self.stem(flux)

        x = self.block1b(self.block1a(x))
        x = self.block2c(self.block2b(self.block2a(x)))
        x = self.block3c(self.block3b(self.block3a(x)))
        x = self.block4c(self.block4b(self.block4a(x)))
        x = self.block5a(x)

        x = x.permute(0, 2, 1) # needed since MultiheadAttention expects (B, L, C)

        attn_out, _ = self.attention(x, x, x)

        x = x + attn_out
        x = x.permute(0, 2, 1)

        x = self.gap(x).squeeze(2)

        if aux is not None and aux.numel() > 0:
            x = torch.cat([x, aux], dim=1)

        return self.classifier(x)