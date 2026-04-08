import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """
    Two-layer 1D residual block with optional downsampling.

    Conv1d -> BN -> ReLU -> Conv1d -> BN
         +-- skip (1x1 conv if channels/stride differ) --+
                            -> ReLU
    """

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

        # Skip connection — only needed when dimensions change
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
    """
    Squeeze-and-Excitation block — lets the network re-weight channels
    based on global context. Cheap but effective for spectral data.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, L)
        w = x.mean(dim=2)                          # global avg pool -> (B, C)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))             # (B, C)
        return x * w.unsqueeze(2)                  # broadcast over L


class SEResidualBlock1D(nn.Module):
    """Residual block + Squeeze-and-Excitation."""

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        self.res = ResidualBlock1D(in_channels, out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride, dropout=dropout)
        self.se  = SEBlock(out_channels)

    def forward(self, x):
        return self.se(self.res(x))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SpectraClassifier(nn.Module):
    """
    1D CNN with 4 SE-Residual blocks for SDSS spectral classification.

    Architecture rationale:
    - Input is a 1D flux array (~4000 wavelength steps).
    - Residual blocks capture local spectral features at multiple scales
      (emission lines are narrow; continuum shape is broad).
    - SE blocks let the network attend to informative channel groups.
    - Strided convolutions downsample progressively instead of pooling,
      preserving more gradient flow.
    - Global average pooling collapses the spatial dimension before the
      classifier head, making the model length-agnostic.
    - Optional auxiliary features (redshift z, VDISP, broadband fluxes)
      are concatenated just before the final FC layer.

    Classes: STAR=0, GALAXY=1, QSO=2
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_length: int = 3522,   # typical SDSS DR17 spectrum length
        aux_features: int = 7,      # Z, VDISP, SPECTROFLUX_U/G/R/I/Z
        dropout: float = 0.3,
    ):
        super().__init__()

        # --- Stem: initial feature extraction ---
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        # After stem: length ~= input_length / 4

        # --- 4 SE-Residual blocks, doubling channels, halving length ---
        self.block1 = SEResidualBlock1D( 32,  64, kernel_size=7, stride=2, dropout=0.1)
        self.block2 = SEResidualBlock1D( 64, 128, kernel_size=7, stride=2, dropout=0.1)
        self.block3 = SEResidualBlock1D(128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.block4 = SEResidualBlock1D(256, 512, kernel_size=3, stride=2, dropout=0.2)

        # --- Global average pooling -> (B, 512) ---
        self.gap = nn.AdaptiveAvgPool1d(1)

        # --- Classifier head ---
        # Concatenate aux features (z, VDISP, broadband fluxes) here
        self.classifier = nn.Sequential(
            nn.Linear(512 + aux_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, flux, aux=None):
        """
        Args:
            flux : (B, L)        raw flux array (will be unsqueezed to (B,1,L))
            aux  : (B, aux_features)  optional auxiliary features
        Returns:
            logits: (B, num_classes)
        """
        x = flux.unsqueeze(1)          # (B, 1, L)

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.gap(x).squeeze(2)    # (B, 512)

        if aux is not None:
            x = torch.cat([x, aux], dim=1)   # (B, 512 + aux_features)
        else:
            # Zero-pad aux slot so the classifier head always gets the same size
            zeros = torch.zeros(x.size(0), self.classifier[0].in_features - 512,
                                device=x.device)
            x = torch.cat([x, zeros], dim=1)

        return self.classifier(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, L = 8, 3522
    model = SpectraClassifier(num_classes=3, input_length=L, aux_features=7)

    flux = torch.randn(B, L)
    aux  = torch.randn(B, 7)

    logits = model(flux, aux)
    print(f"Input flux:  {flux.shape}")
    print(f"Input aux:   {aux.shape}")
    print(f"Output logits: {logits.shape}")   # (8, 3)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")