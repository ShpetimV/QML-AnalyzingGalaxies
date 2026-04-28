"""
Quantum ML Model for SDSS Spectral Classification
====================================================
Angle encoding with data-reuploading VQC:

  1. Heavy classical feature extraction
       - 3 conv blocks with progressive channel growth (1 → 16 → 32 → 64)
       - 3-stage MLP funnel (256 → 32 → 16 → out_features)
       - Smaller final feature dimension (default 4)

  2. Data-reuploading VQC
       - 4 qubits (matches feature dim)
       - 6 re-uploading layers for expressivity
       - Circular CNOT entanglement, RY angle encoding

Batch format from the dataloader:
    batch['flux']  : [B, 1, L]   (L = 4096 after crop/pad)
    batch['label'] : [B]
"""

import math

import pennylane as qml
import torch
import torch.nn as nn


class SpectralFeatureExtractor(nn.Module):
    """
    Heavy 1D CNN that compresses ~4096 spectral bins down to n_qubits features.

    Architecture:
      Block 1: Conv(1→16,  k=15, s=4) → BN → ReLU → MaxPool(2)
      Block 2: Conv(16→32, k=9,  s=2) → BN → ReLU → MaxPool(2)
      Block 3: Conv(32→64, k=5,  s=2) → BN → ReLU → MaxPool(2)
      Head   : AdaptiveAvgPool(4) → Flatten
                → Linear(256→32) → ReLU
                → Linear(32→16)  → ReLU
                → Linear(16→out_features)
    """

    def __init__(self, out_features: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.AdaptiveAvgPool1d(4),
        )
        self.proj = nn.Sequential(
            nn.Linear(64 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_features),
        )

    def forward(self, x):
        """x: [B, 1, L] → [B, out_features]"""
        h = self.backbone(x)
        return self.proj(h.reshape(x.size(0), -1))


class AngleEncodingClassifier(nn.Module):
    """
    Hybrid quantum-classical model with angle (rotation) encoding.

    Pipeline:
      flux → heavy CNN extractor → small feature vector (dim = n_qubits)
           → bounded to [-π, π] via tanh
           → data-reuploading VQC (RY encoding + Rot gates + CNOT ring)
           → PauliZ expectation values
           → MLP head → logits
    """

    def __init__(
            self,
            num_classes: int = 2,
            n_qubits: int = 4,
            n_layers: int = 6,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.extractor = SpectralFeatureExtractor(out_features=n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(
            self._circuit, self.dev,
            interface="torch", diff_method="backprop",
        )

        # (n_layers, n_qubits, 3) — Rot gate params per layer per qubit
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        self.head = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def _circuit(self, features, weights):
        """
        Data-reuploading VQC.

          features : (B, n_qubits) — PennyLane broadcasts over batch
          weights  : (n_layers, n_qubits, 3) — shared across batch
        """
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RY(features[:, q], wires=q)
            for q in range(self.n_qubits):
                qml.Rot(
                    weights[layer, q, 0],
                    weights[layer, q, 1],
                    weights[layer, q, 2],
                    wires=q,
                )
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def forward(self, flux, *_):
        """
        Args:
            flux : [B, 1, L]
        Returns:
            logits : [B, num_classes]
        """
        original_device = flux.device

        feat = self.extractor(flux)
        feat = torch.tanh(feat) * math.pi

        # PennyLane requires CPU tensors
        q_list = self.qnode(feat.cpu(), self.q_weights.cpu())
        q_out = torch.stack(q_list, dim=1).float().to(original_device)

        return self.head(q_out)
