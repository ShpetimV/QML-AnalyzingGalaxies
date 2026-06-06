"""
Quantum Convolutional Neural Network (QCNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

N_QUBITS = 12

def conv_block(p, w0, w1):
    qml.SpecialUnitary(p, wires=[w0, w1])   # full SU(4), 4^2 - 1 = 15 params


def pool_block(p, source, sink):
    """Coherent pooling, 2 params."""
    qml.CRZ(p[0], wires=[source, sink])
    qml.PauliX(wires=source)
    qml.CRX(p[1], wires=[source, sink])


def conv_layer(p, wires):
    """Brickwork of the SHARED conv_block over neighbouring pairs + wrap."""
    n = len(wires)
    for i in range(0, n - 1, 2):
        conv_block(p, wires[i], wires[i + 1])
    for i in range(1, n - 1, 2):
        conv_block(p, wires[i], wires[i + 1])
    if n > 2:
        conv_block(p, wires[-1], wires[0])


def pool_layer(p, sources, sinks):
    for s, t in zip(sources, sinks):
        pool_block(p, s, t)


def _readout_ops(survivors, measure, add_zz=False):
    ops = []
    for w in survivors:
        for m in measure:
            if m == "Z":
                ops.append(qml.expval(qml.PauliZ(w)))
            elif m == "X":
                ops.append(qml.expval(qml.PauliX(w)))
            elif m == "Y":
                ops.append(qml.expval(qml.PauliY(w)))
    if add_zz:
        # Pairwise correlators capture entanglement between survivors.
        for i in range(len(survivors)):
            for j in range(i + 1, len(survivors)):
                wi, wj = survivors[i], survivors[j]
                ops.append(qml.expval(qml.PauliZ(wi) @ qml.PauliZ(wj)))
    return ops


# --------------------------------------------------------------------------- #
#  Amplitude-encoded QCNN                                                     #
# --------------------------------------------------------------------------- #
class QCNNClassifier(nn.Module):
    """n_qubits=12 default. Amplitude-encoded, so 4096 features max (zero-padded if <4096)."""

    def __init__(self, num_classes=4, n_qubits=N_QUBITS,
                 device_name="default.qubit", measure=("Z", "X", "Y")):
        super().__init__()
        self.n_qubits = n_qubits
        self.measure = tuple(measure)

        # weight-shared variational params
        self.conv_w = nn.Parameter(0.1 * torch.randn(3, 15))
        self.pool_w = nn.Parameter(0.1 * torch.randn(2, 2))

        self.survivors = [3, 7, 11]
        n_pairs = len(self.survivors) * (len(self.survivors) - 1) // 2
        n_feat = len(self.survivors) * len(self.measure) + n_pairs
        self.feat_norm = nn.BatchNorm1d(n_feat, affine=False)
        self.readout = nn.Linear(n_feat, num_classes)

        self.dev = qml.device(device_name, wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev,
                               interface="torch", diff_method="backprop")

    def _circuit(self, features, conv_w, pool_w):
        qml.AmplitudeEmbedding(features, wires=range(self.n_qubits),
                               normalize=True, pad_with=0.0)
        # Layer 1: conv on 12, pool 12 -> 6
        conv_layer(conv_w[0], list(range(12)))
        pool_layer(pool_w[0], [0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11])
        # Layer 2: conv on 6, pool 6 -> 3
        conv_layer(conv_w[1], [1, 3, 5, 7, 9, 11])
        pool_layer(pool_w[1], [1, 5, 9], [3, 7, 11])
        # Layer 3: deep conv on the 3 survivors
        conv_layer(conv_w[2], [3, 7, 11])
        return _readout_ops(self.survivors, self.measure, add_zz=True)

    def forward(self, flux, aux=None):
        x = flux.squeeze(1) if flux.dim() == 3 else flux
        if x.shape[-1] > 4096:
            x = x[..., :4096]
        elif x.shape[-1] < 4096:
            x = F.pad(x, (0, 4096 - x.shape[-1]))
        x = F.normalize(x, p=2, dim=-1)

        dev_in = x.device  # default.qubit statevector sim runs on CPU
        q_out = self.qnode(x.cpu(), self.conv_w.cpu(), self.pool_w.cpu())
        q_out = torch.stack(q_out, dim=-1).float().to(dev_in)
        q_out = self.feat_norm(q_out)
        return self.readout(q_out)


# --------------------------------------------------------------------------- #
#  Hybrid: classical compression -> angle encoding -> same quantum conv/pool  #
# --------------------------------------------------------------------------- #
class HybridQCNNClassifier(nn.Module):
    """n_qubits=12 default. Amplitude-encoded, so 4096 features max (zero-padded if <4096)."""

    def __init__(self, num_classes=4, n_qubits=10,
                 device_name="default.qubit", measure=("Z", "X", "Y"), reuploads=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.measure = tuple(measure)
        self.reuploads = reuploads

        # Split stem so adaptive pooling can run on CPU (MPS limitation on non-divisible sizes).
        self.stem_conv = nn.Conv1d(1, 8, kernel_size=21, stride=8, padding=10)
        self.stem_act = nn.ReLU()
        self.stem_mix = nn.Conv1d(8, 1, kernel_size=1)  # mixes channels, NOT positions

        self.conv_w = nn.Parameter(0.1 * torch.randn(reuploads, 2, 15))
        self.pool_w = nn.Parameter(0.1 * torch.randn(2, 2))

        self.survivors = [3, 7, 9]
        n_pairs = len(self.survivors) * (len(self.survivors) - 1) // 2
        n_feat = len(self.survivors) * len(self.measure) + n_pairs
        self.feat_norm = nn.BatchNorm1d(n_feat, affine=False)
        self.readout = nn.Linear(n_feat, num_classes)

        self.dev = qml.device(device_name, wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev,
                               interface="torch", diff_method="backprop")

    def _circuit(self, feats, conv_w, pool_w):
        wires = list(range(self.n_qubits))
        for r in range(self.reuploads):
            qml.AngleEmbedding(feats, wires=wires, rotation="Y")  # data re-upload
            conv_layer(conv_w[r, 0], wires)
            conv_layer(conv_w[r, 1], wires)
        pool_layer(pool_w[0], [0, 2, 4, 6, 8], [1, 3, 5, 7, 9])  # 10 -> 5
        pool_layer(pool_w[1], [1, 5], [3, 7])                   # 5 -> 3 (keep 9)
        return _readout_ops(self.survivors, self.measure, add_zz=True)

    def forward(self, flux, aux=None):
        x = flux if flux.dim() == 3 else flux.unsqueeze(1)
        x = self.stem_act(self.stem_conv(x))
        # MPS limitation: adaptive_avg_pool1d requires divisible sizes; run on CPU.
        pooled = F.adaptive_avg_pool1d(x.cpu(), self.n_qubits).to(x.device)
        feats = self.stem_mix(pooled).flatten(1)  # [B, n_qubits]
        feats = torch.tanh(feats) * torch.pi       # keep angles in (-pi, pi)
        dev_in = feats.device
        q_out = self.qnode(feats.cpu(), self.conv_w.cpu(), self.pool_w.cpu())
        q_out = torch.stack(q_out, dim=-1).float().to(dev_in)
        q_out = self.feat_norm(q_out)
        return self.readout(q_out)


# --------------------------------------------------------------------------- #
#  Classical baselines                                                        #
# --------------------------------------------------------------------------- #
class ClassicalTinyBaseline(nn.Module):
    """matches the number of trainable params of the QCNNClassifier -> 103"""
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=5, stride=4, padding=2)
        self.conv2 = nn.Conv1d(4, 3, kernel_size=3, stride=4, padding=1)
        self.pool  = nn.AdaptiveMaxPool1d(3)
        self.fc    = nn.Linear(9, num_classes)
    def forward(self, flux, aux=None):
        if flux.shape[-1] > 4096: flux = flux[:, :, :4096]
        elif flux.shape[-1] < 4096: flux = F.pad(flux, (0, 4096 - flux.shape[-1]))
        x = F.relu(self.conv1(flux)); x = F.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))

class ClassicalHugeBaseline(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.AdaptiveMaxPool1d(4),
            nn.Flatten(),
            nn.Linear(32 * 4, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, flux, aux=None):
        if flux.shape[-1] > 4096:
            flux = flux[:, :, :4096]
        elif flux.shape[-1] < 4096:
            flux = F.pad(flux, (0, 4096 - flux.shape[-1]))
        return self.net(flux)
