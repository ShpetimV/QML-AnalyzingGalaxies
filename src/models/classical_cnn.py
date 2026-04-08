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
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(2)


class SEResidualBlock1D(nn.Module):
    """Residual block with 1D convolutions and Squeeze-and-Excitation."""

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2

        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels)
        )

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
        return F.relu(self.se(out))


class SpectraClassifier(nn.Module):
    """Refined 1D SE-ResNet for SDSS classification."""

    def __init__(self, num_classes=3, aux_features=6, dropout=0.3):
        super().__init__()

        # Initial feature extraction (Stem)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # SE-Residual Blocks
        self.block1 = SEResidualBlock1D(32, 64, kernel_size=7, stride=2, dropout=0.1)
        self.block2 = SEResidualBlock1D(64, 128, kernel_size=7, stride=2, dropout=0.1)
        self.block3 = SEResidualBlock1D(128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.block4 = SEResidualBlock1D(256, 512, kernel_size=3, stride=2, dropout=0.2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head (handles flux features + optional scalar data)
        self.classifier = nn.Sequential(
            nn.Linear(512 + aux_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, flux, aux=None):
        # Handle input shape from new DataLoader: [Batch, 1, Length]
        if flux.dim() == 2:
            flux = flux.unsqueeze(1)

        x = self.stem(flux)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).squeeze(2)

        if aux is not None and aux.numel() > 0:
            x = torch.cat([x, aux], dim=1)

        return self.classifier(x)