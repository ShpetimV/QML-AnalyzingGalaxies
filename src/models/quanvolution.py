"""
Experiment 2 Models: Quanvolution vs. Classical Convolution
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

LIGHT_HEAD_DIAGNOSTIC = False
import os
QUANTUM_DEVICE = os.environ.get("QUANTUM_DEVICE", "default.qubit")

class Quanv1dLayer(nn.Module):
    """
    Extreme Quanvolution with Bounded Attention and a Trainable Quantum Lens.
    """

    def __init__(self, kernel_size=4, stride=2, n_layers=3, trainable=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_qubits = kernel_size
        self.n_layers = n_layers

        # 1. Standard rotation weights (36 params)
        self.q_weights = nn.Parameter(torch.randn(n_layers, self.n_qubits, 3) * 0.1)

        # 2. Bounded Quantum Attention (4 params) — init near pass-through, not 0.5
        self.input_squeeze = nn.Parameter(torch.full((self.n_qubits,), 2.0))

        # 3. UPGRADE: The Quantum Lens (12 params)
        self.lens_weights = nn.Parameter(torch.randn(self.n_qubits, 3) * 0.1)

        if not trainable:
            self.q_weights.requires_grad = False
            self.input_squeeze.requires_grad = False
            self.lens_weights.requires_grad = False

        if QUANTUM_DEVICE == "lightning.qubit":
            # Tries to claim the NVIDIA GPU (Ubuntu Server)
            self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
            self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="adjoint")
        else:
            # Falls back to Apple Silicon CPU / Standard CPU (MacBook)
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
            self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")

        print(f"Successfully initialized Quanv1dLayer with {self.n_qubits} qubits on device: {self.dev.name}")

    def _circuit(self, inputs, weights, lens_weights):
        for layer in range(self.n_layers):
            # Multi-Axis DRU
            for i in range(self.n_qubits):
                if layer % 3 == 0:
                    qml.RY(inputs[:, i], wires=i)
                elif layer % 3 == 1:
                    qml.RZ(inputs[:, i], wires=i)
                else:
                    qml.RX(inputs[:, i], wires=i)

            # Standard Ring ZZ (Robust Entanglement)
            for i in range(self.n_qubits - 1):
                qml.IsingZZ(inputs[:, i] * inputs[:, i + 1], wires=[i, i + 1])
            qml.IsingZZ(inputs[:, -1] * inputs[:, 0], wires=[self.n_qubits - 1, 0])

            # Entanglement Rotations
            for i in range(self.n_qubits):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        # THE QUANTUM LENS: Final rotation to align the state away from noise
        for i in range(self.n_qubits):
            qml.Rot(lens_weights[i, 0], lens_weights[i, 1], lens_weights[i, 2], wires=i)

        measurements = []
        for i in range(self.n_qubits):  # 4 local
            measurements.append(qml.expval(qml.PauliZ(i)))
        for i in range(self.n_qubits):  # 4 local
            measurements.append(qml.expval(qml.PauliX(i)))
        for i in range(self.n_qubits):  # 4 two-body correlations
            measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 1) % self.n_qubits)))
        return measurements

    def forward(self, x):
        original_device = x.device
        batch_size, _, seq_len = x.shape

        patches = x.unfold(dimension=2, size=self.kernel_size, step=self.stride)
        num_patches = patches.size(2)
        patches_flat = patches.reshape(-1, self.kernel_size)

        # Spherical Encoding (Max pi/2)
        patch_norms = torch.norm(patches_flat, p=2, dim=1, keepdim=True) + 1e-8
        patches_squashed = (patches_flat / patch_norms) * (math.pi / 2.0)

        # Apply Bounded Attention
        attention_mask = torch.sigmoid(self.input_squeeze)
        patches_attended = patches_squashed * attention_mask

        # Execute with the Lens weights
        with torch.autocast(device_type=original_device.type, enabled=False):
            q_results = self.qnode(patches_attended, self.q_weights, self.lens_weights)

        out_channels = self.n_qubits * 3
        out_flat = torch.stack(q_results, dim=1).float().to(original_device)
        out = out_flat.view(batch_size, num_patches, out_channels).permute(0, 2, 1)
        return out


class QuanvClassifier(nn.Module):
    def __init__(self, seq_len=1024, kernel_size=4, stride=2, n_layers=3, trainable_filter=True, num_classes=2):
        super().__init__()
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = kernel_size * 3 # 12

        # Pure Quantum Filter
        self.quanv = Quanv1dLayer(kernel_size, stride, n_layers, trainable_filter)

        if LIGHT_HEAD_DIAGNOSTIC:
            # Pure Aggregation
            self.agg = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )

            self.classifier = nn.Linear(self.out_channels, num_classes)
        else:
            # Deep Aggregation
            self.agg = nn.Sequential(
                nn.Conv1d(self.out_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1), # Kept the MaxPool spotlight!
                nn.Flatten()
            )

            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )

    def forward(self, flux, aux=None):
        flux_pooled = F.adaptive_avg_pool1d(flux, self.seq_len)
        q_features = self.quanv(flux_pooled)
        pooled_features = self.agg(q_features)
        return self.classifier(pooled_features)


class ClassicalConvClassifier(nn.Module):
    def __init__(self, seq_len=1024, kernel_size=4, stride=2, trainable_filter=True, num_classes=2):
        super().__init__()
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = kernel_size * 3 # 12

        h = 2
        self.filter = nn.Sequential(
            nn.Linear(kernel_size, h), nn.Tanh(),
            nn.Linear(h, self.out_channels), nn.Tanh()  # final Tanh to bound like Pauli expvals [-1,1]
        )
        if not trainable_filter:
            for p in self.filter.parameters():
                p.requires_grad = False

        if LIGHT_HEAD_DIAGNOSTIC:
            self.agg = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            self.classifier = nn.Linear(self.out_channels, num_classes)

        else:
            # Deep Aggregation
            self.agg = nn.Sequential(
                nn.Conv1d(self.out_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten()
            )

            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )

    def forward(self, flux, aux=None):
        flux_pooled = F.adaptive_avg_pool1d(flux, self.seq_len)

        patches = flux_pooled.unfold(dimension=2, size=self.kernel_size, step=self.stride)
        batch_size = patches.size(0)
        num_patches = patches.size(2)
        patches_flat = patches.reshape(-1, self.kernel_size)

        # Spherical Encoding matches Quantum exactly
        patch_norms = torch.norm(patches_flat, p=2, dim=1, keepdim=True) + 1e-8
        patches_squashed = (patches_flat / patch_norms) * (math.pi / 2.0)

        # Apply Linear Filter
        c_features = self.filter(patches_squashed)
        c_features = c_features.view(batch_size, num_patches, self.out_channels).permute(0, 2, 1)

        pooled_features = self.agg(c_features)
        return self.classifier(pooled_features)