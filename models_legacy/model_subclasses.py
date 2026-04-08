import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return F.relu(out)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        w = x.mean(dim=2)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(2)


class SEResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        self.res = ResidualBlock1D(in_channels, out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride, dropout=dropout)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        return self.se(self.res(x))


class SpectraClassifier(nn.Module):
    """
    1D SE-ResNet for SDSS spectral classification.
    num_classes is passed dynamically from train.py based on the
    actual subclass count after grouping and filtering.
    """

    def __init__(
        self,
        num_classes: int = 24,
        input_length: int = 3522,
        aux_features: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.block1 = SEResidualBlock1D( 32,  64, kernel_size=7, stride=2, dropout=0.1)
        self.block2 = SEResidualBlock1D( 64, 128, kernel_size=7, stride=2, dropout=0.1)
        self.block3 = SEResidualBlock1D(128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.block4 = SEResidualBlock1D(256, 512, kernel_size=3, stride=2, dropout=0.2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        # Hidden layer widened to 128 to handle more output classes
        self.classifier = nn.Sequential(
            nn.Linear(512 + aux_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, flux, aux=None):
        x = flux.unsqueeze(1)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).squeeze(2)

        if aux is not None:
            x = torch.cat([x, aux], dim=1)
        else:
            zeros = torch.zeros(x.size(0), self.classifier[0].in_features - 512,
                                device=x.device)
            x = torch.cat([x, zeros], dim=1)

        return self.classifier(x)


if __name__ == "__main__":
    model = SpectraClassifier(num_classes=24, input_length=3522, aux_features=0)
    flux  = torch.randn(8, 3522)
    out   = model(flux)
    print(f"Output shape: {out.shape}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")