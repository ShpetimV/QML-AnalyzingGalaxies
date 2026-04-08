"""
Quantum ML Models for SDSS Spectral Classification
====================================================
Two encoding strategies, both using PennyLane + PyTorch:

  1. AngleEncodingClassifier
     - Classical CNN compresses ~4448 flux bins → n_qubits features
     - Features encoded as RY rotation angles
     - Data-reuploading variational circuit
     - Measurements → classical head → 3-class logits

  2. AmplitudeEncodingClassifier
     - Classical CNN compresses flux → 2^n_qubits features
     - L2-normalised vector → amplitude-encoded quantum state
     - Variational layers → measurements → head → logits
     - Much richer encoding (exponential state space) but slower

Both accept the dataloader format:
    batch['flux']    : [B, 1, L]   (L ≈ 4448 after crop)
    batch['scalars'] : [B, 6]      (Z, SPECTROFLUX_U/G/R/I/Z)
    batch['label']   : [B]

Classes: STAR=0, GALAXY=1, QSO=2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import math


# ---------------------------------------------------------------------------
# Classical Feature Extractor (shared)
# ---------------------------------------------------------------------------

class SpectralFeatureExtractor(nn.Module):
    """
    Small 1D CNN that compresses ~4448 spectral bins into `out_features`
    values, preserving spectral shape better than naive avg-pooling.

    Architecture:
      Conv(1→16, k=15, s=4) → BN → ReLU → MaxPool(4)
      Conv(16→32, k=7, s=2) → BN → ReLU → AdaptiveAvgPool → Flatten
      Linear → ReLU → Linear(out_features)
    """

    def __init__(self, out_features: int = 8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),   # → [B, 32, 8]
        )
        self.proj = nn.Sequential(
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x):
        """x: [B, 1, L] → [B, out_features]"""
        B = x.size(0)
        h = self.backbone(x)          # [B, 32, 8]
        h = h.reshape(B, -1)          # [B, 256]
        return self.proj(h)            # [B, out_features]


# ---------------------------------------------------------------------------
# 1. Angle Encoding
# ---------------------------------------------------------------------------

class AngleEncodingClassifier(nn.Module):
    """
    Hybrid quantum-classical model with angle (rotation) encoding.

    Each reduced feature is mapped to a qubit via RY(feature * π).
    Data-reuploading: features are re-encoded every variational layer,
    giving the circuit the expressivity of a Fourier series.
    """

    def __init__(
        self,
        num_classes: int = 3,
        n_qubits: int = 8,
        n_layers: int = 4,
        n_scalars: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical feature reduction: flux → n_qubits values
        self.extractor = SpectralFeatureExtractor(out_features=n_qubits)

        # Quantum device + circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev,
                               interface="torch", diff_method="backprop")

        # Trainable quantum parameters: (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        # Classical post-processing
        head_in = n_qubits + n_scalars
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def _circuit(self, features, weights):
        """
        Data-reuploading VQC with angle encoding.
        Supports PennyLane parameter broadcasting:
          features : (B, n_qubits) — batched input
          weights  : (n_layers, n_qubits, 3) — shared across batch
        Each features[:, q] is shape (B,) → PennyLane executes
        the entire batch in one vectorised pass (no Python loop).
        """
        for layer in range(self.n_layers):
            # Encode data as RY rotations (re-uploaded each layer)
            for q in range(self.n_qubits):
                qml.RY(features[:, q], wires=q)    # (B,) — broadcasted

            # Trainable rotations (scalars — shared across batch)
            for q in range(self.n_qubits):
                qml.Rot(weights[layer, q, 0],
                        weights[layer, q, 1],
                        weights[layer, q, 2], wires=q)

            # Entanglement: circular CNOT chain
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def forward(self, flux, scalars=None):
        """
        Args:
            flux    : [B, 1, L]
            scalars : [B, n_scalars] or None
        Returns:
            logits  : [B, num_classes]
        """
        B = flux.size(0)

        # Extract + scale features to [-π, π]
        feat = self.extractor(flux)                   # [B, n_qubits]
        feat = torch.tanh(feat) * math.pi             # bound to [-π, π]

        # Run quantum circuit — entire batch at once via broadcasting
        q_list = self.qnode(feat, self.q_weights)     # list of n_qubits tensors, each (B,)
        q_out = torch.stack(q_list, dim=1).float()    # [B, n_qubits]

        # Concatenate scalar features
        if scalars is not None and scalars.numel() > 0:
            q_out = torch.cat([q_out, scalars], dim=1)
        else:
            pad = torch.zeros(B, self.head[0].in_features - self.n_qubits,
                              device=flux.device)
            q_out = torch.cat([q_out, pad], dim=1)

        return self.head(q_out)


# ---------------------------------------------------------------------------
# 2. Amplitude Encoding
# ---------------------------------------------------------------------------

class AmplitudeEncodingClassifier(nn.Module):
    """
    Hybrid model with amplitude encoding.

    The classical extractor reduces flux to 2^n_qubits features,
    which are L2-normalised into a valid quantum state vector and
    loaded via StatePrep (amplitude encoding).

    This encodes exponentially more information per qubit than angle
    encoding, but state preparation is deeper and simulation is slower.

    Default: 8 qubits → 256-dim state vector (practical for simulation).
    For the full 13-qubit (8192-dim) version, pass n_qubits=13 and
    set use_classical_reduction=False, but expect ~100× slower training.
    """

    def __init__(
        self,
        num_classes: int = 3,
        n_qubits: int = 8,
        n_layers: int = 3,
        n_scalars: int = 6,
        dropout: float = 0.2,
        use_classical_reduction: bool = True,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_dim = 2 ** n_qubits
        self.use_classical_reduction = use_classical_reduction

        # Classical feature extractor → 2^n_qubits features
        if use_classical_reduction:
            self.extractor = SpectralFeatureExtractor(out_features=self.state_dim)

        # Quantum device + circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev,
                               interface="torch", diff_method="backprop")

        # Trainable quantum parameters
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        # Classical head
        head_in = n_qubits + n_scalars
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def _circuit(self, state_vector, weights):
        """
        Amplitude-encoded VQC.
          state_vector : (2^n_qubits,) normalised amplitudes
          weights      : (n_layers, n_qubits, 3)
        """
        # Amplitude encoding via state preparation
        qml.StatePrep(state_vector, wires=range(self.n_qubits))

        # Variational layers
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.Rot(weights[layer, q, 0],
                        weights[layer, q, 1],
                        weights[layer, q, 2], wires=q)
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def _prepare_state(self, flux):
        """
        Prepare a valid quantum state vector from flux data.
        Returns: [B, 2^n_qubits] L2-normalised vectors.
        """
        B = flux.size(0)

        if self.use_classical_reduction:
            # Reduce classically to state_dim features
            x = self.extractor(flux)                           # [B, state_dim]
        else:
            # Use raw flux: squeeze channel, pad to 2^n_qubits
            x = flux.squeeze(1)                                # [B, L]
            pad_size = self.state_dim - x.shape[-1]
            if pad_size > 0:
                x = F.pad(x, (0, pad_size))
            elif pad_size < 0:
                x = x[:, :self.state_dim]

        # L2 normalise → valid quantum state
        x = F.normalize(x, p=2, dim=-1)

        # Guard against zero-norm edge case
        norms = x.norm(dim=-1, keepdim=True)
        zero_mask = (norms < 1e-8).squeeze(-1)
        if zero_mask.any():
            uniform = torch.ones(self.state_dim, device=x.device) / math.sqrt(self.state_dim)
            x[zero_mask] = uniform

        return x

    def forward(self, flux, scalars=None):
        """
        Args:
            flux    : [B, 1, L]
            scalars : [B, n_scalars] or None
        Returns:
            logits  : [B, num_classes]
        """
        B = flux.size(0)

        state_vectors = self._prepare_state(flux)   # [B, 2^n_qubits]

        # Quantum circuit per sample
        # NOTE: StatePrep doesn't support parameter broadcasting,
        # so amplitude encoding must loop per-sample (slower than angle).
        q_out = torch.stack([
            torch.stack(self.qnode(state_vectors[i], self.q_weights))
            for i in range(B)
        ]).float()  # [B, n_qubits] — cast from float64 to float32

        # Concatenate scalars
        if scalars is not None and scalars.numel() > 0:
            q_out = torch.cat([q_out, scalars], dim=1)
        else:
            pad = torch.zeros(B, self.head[0].in_features - self.n_qubits,
                              device=flux.device)
            q_out = torch.cat([q_out, pad], dim=1)

        return self.head(q_out)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_quantum_model(encoding: str = "angle", **kwargs) -> nn.Module:
    """
    Factory:
      encoding = "angle"     → AngleEncodingClassifier
      encoding = "amplitude" → AmplitudeEncodingClassifier
    """
    if encoding == "angle":
        return AngleEncodingClassifier(**kwargs)
    elif encoding == "amplitude":
        return AmplitudeEncodingClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown encoding '{encoding}'. Use 'angle' or 'amplitude'.")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ANGLE ENCODING — 8 qubits, 4 layers")
    print("=" * 60)
    m1 = AngleEncodingClassifier(num_classes=3, n_qubits=8, n_layers=4, n_scalars=6)
    x = torch.randn(2, 1, 4448)
    s = torch.randn(2, 6)
    out1 = m1(x, s)
    p1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    print(f"  flux {x.shape}  scalars {s.shape}  → logits {out1.shape}")
    print(f"  Trainable params: {p1:,}")

    print()
    print("=" * 60)
    print("AMPLITUDE ENCODING — 8 qubits (256-dim state), 3 layers")
    print("=" * 60)
    m2 = AmplitudeEncodingClassifier(num_classes=3, n_qubits=8, n_layers=3, n_scalars=6)
    out2 = m2(x, s)
    p2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"  flux {x.shape}  scalars {s.shape}  → logits {out2.shape}")
    print(f"  Trainable params: {p2:,}")