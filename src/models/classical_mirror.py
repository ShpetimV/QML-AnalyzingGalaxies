"""
Classical Mirror Model — dequantized counterpart to AngleEncodingClassifier
=============================================================================
Designed for a fair apples-to-apples comparison with the quantum model.

Architecture is identical in every way except the quantum circuit is replaced
by a small classical MLP with the same parameter count (~72 params):

  Quantum model:  extractor → tanh*π → VQC (72 params, 4q×6L) → head
  This model:     extractor → tanh*π → MLP (76 params, 4→8→4)  → head

The extractor and head are bit-for-bit identical to AngleEncodingClassifier.
The only difference is the quantum circuit vs the classical MLP in the middle.
"""

import math

import torch
import torch.nn as nn

from src.models.quantum_model import SpectralFeatureExtractor


class ClassicalMirrorClassifier(nn.Module):
    """
    Dequantized mirror of AngleEncodingClassifier.

    Replace VQC (4 qubits × 6 layers × 3 params = 72 params) with a
    2-layer MLP of equal parameter count: Linear(4→8)→ReLU→Linear(8→4) = 76 params.

    Everything else — extractor, tanh*π preprocessing, and head — is identical
    to the quantum model so performance differences isolate the quantum vs
    classical processing step.
    """

    def __init__(
            self,
            num_classes: int = 2,
            n_features: int = 4,
            dropout: float = 0.2,
    ):
        super().__init__()

        # Exact same extractor as AngleEncodingClassifier
        self.extractor = SpectralFeatureExtractor(out_features=n_features)

        # Classical replacement for the VQC
        # Linear(4→8): 4*8+8 = 40 params
        # Linear(8→4): 8*4+4 = 36 params  → total 76 ≈ 72 quantum params
        self.classical_layer = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.ReLU(),
            nn.Linear(8, n_features),
        )

        # Exact same head as AngleEncodingClassifier
        self.head = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, flux, *_):
        """
        Args:
            flux : [B, 1, L]
        Returns:
            logits : [B, num_classes]
        """
        feat = self.extractor(flux)
        feat = torch.tanh(feat) * math.pi  # identical preprocessing to quantum model
        feat = self.classical_layer(feat)
        return self.head(feat)
