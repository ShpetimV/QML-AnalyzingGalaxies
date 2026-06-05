"""
Experiment 3: Complex Correlation Showdown — VQC vs Classical Dense Network
=============================================================================
Both pipelines share a FROZEN pretrained SpectraClassifier ("the Beast") used
purely as a feature extractor. The final Linear(128, num_classes) layer of the
Beast is replaced with Identity, so the extractor outputs 128-dim features.

The two contenders differ only in what processes those features:

  Classical Dense head:  Linear(128, 38) → ReLU → Dropout → Linear(38, 4)
                         ≈ 5058 trainable parameters

  Quantum VQC head:      frozen PCA projection (128 → n_qubits)   [non-trainable buffer]
                       → tanh(·) * π
                       → data-reuploading VQC (n_layers × n_qubits × 1 RY)
                       → Linear(n_qubits, num_classes) trainable readout
                         ≈ 40 trainable parameters (4q × 5L VQC + 4×4+4 readout)

The PCA projection picks the top-n_qubits axes of variance from the Beast's
feature space — frozen, but informative (unlike a random projection).
The trainable readout decouples "qubit i means class i" and lets logits
escape the [-1, 1] expval range so softmax confidence isn't capped.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sklearn.decomposition import PCA

from src.models.classical_cnn import SpectraClassifier


# ---------------------------------------------------------------------------
# Frozen PCA bottleneck — shared by the PCA variants below
# ---------------------------------------------------------------------------

class FrozenPCABottleneck(nn.Module):
    """
    PCA(in_dim → out_dim), fit once on training-set features, then frozen.

    Stores components/mean as torch buffers so they live on-device with the
    rest of the model and serialize naturally with state_dict. The 'fitted'
    flag is also a buffer (0-d bool tensor) so reloading a checkpoint
    restores it without manual flag-setting.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.register_buffer("components", torch.zeros(out_dim, in_dim))
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, features) -> None:
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        pca = PCA(n_components=self.out_dim, random_state=42)
        pca.fit(features)
        target_device = self.components.device
        self.components.copy_(torch.from_numpy(pca.components_).float().to(target_device))
        self.mean.copy_(torch.from_numpy(pca.mean_).float().to(target_device))
        self.fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("FrozenPCABottleneck.fit() must be called before forward.")
        return (x - self.mean) @ self.components.T


def _fit_pca_on_loader(extractor, bottleneck, loader, device, max_batches=None):
    """Run training data through the frozen extractor and fit PCA on collected features."""
    extractor.to(device).eval()
    feats_chunks = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            flux = batch['flux'].to(device)
            aux = batch.get('aux')
            if aux is not None:
                aux = aux.to(device)
            feats_chunks.append(extractor(flux, aux).cpu())
    feats = torch.cat(feats_chunks, dim=0)
    bottleneck.fit(feats)


# ---------------------------------------------------------------------------
# Frozen pretrained Beast — shared by both contenders
# ---------------------------------------------------------------------------

class FrozenBeastExtractor(nn.Module):
    """
    Loads a pretrained SpectraClassifier checkpoint, replaces its final
    Linear(128, num_classes) layer with Identity, and freezes every parameter.

    Output: 128-dim feature vector per spectrum.

    The Beast is permanently in eval mode so its BatchNorm running stats do
    not drift during downstream training.
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if not isinstance(state, dict) or 'classifier.0.weight' not in state:
            raise ValueError(f"Checkpoint at {checkpoint_path} is not a SpectraClassifier state_dict")

        # Auto-detect the original num_classes and aux_features from checkpoint shapes.
        # Sort numerically — 'classifier.10' comes after 'classifier.9', not after 'classifier.1'.
        beast_aux_features = state['classifier.0.weight'].shape[1] - 1024
        cls_keys = [k for k in state.keys()
                    if k.startswith('classifier.') and k.endswith('.weight')]
        last_key = max(cls_keys, key=lambda k: int(k.split('.')[1]))
        beast_num_classes = state[last_key].shape[0]

        self.beast = SpectraClassifier(num_classes=beast_num_classes, aux_features=beast_aux_features)
        self.beast.load_state_dict(state)

        # Replace the final Linear(128, num_classes) with Identity → 128-dim features
        self.beast.classifier[-1] = nn.Identity()

        for p in self.beast.parameters():
            p.requires_grad = False
        self.beast.eval()

    def train(self, mode: bool = True):
        # Keep the beast in eval mode no matter what the parent does
        super().train(mode)
        self.beast.eval()
        return self

    def forward(self, flux, aux=None):
        with torch.no_grad():
            return self.beast(flux, aux)


# ---------------------------------------------------------------------------
# Classical Dense head — ~5000 trainable params
# ---------------------------------------------------------------------------

class FrozenBeastDenseClassifier(nn.Module):
    """
    Frozen Beast → 2-layer MLP → logits.

    Linear(128, 38): 128*38 + 38 = 4902
    Linear(38, 4):   38*4 + 4    = 156
    Total trainable: 5058
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 4,
                 feature_dim: int = 128, hidden_dim: int = 38, dropout: float = 0.2):
        super().__init__()
        self.extractor = FrozenBeastExtractor(checkpoint_path)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, flux, *_):
        feats = self.extractor(flux)
        return self.head(feats)


# ---------------------------------------------------------------------------
# Quantum VQC head — ~550 trainable params (Hybrid Bottleneck approach)
# ---------------------------------------------------------------------------

class FrozenBeastVQCClassifier(nn.Module):
    """
    Frozen Beast → Trainable Bottleneck (128 → n_qubits) [~516 trainable params]
        → tanh * π
        → data-reuploading VQC (n_qubits × n_layers × 1 RY = 20 trainable)
        → Linear(n_qubits, num_classes) trainable readout (20 trainable params)
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 4,
                 n_qubits: int = 4, n_layers: int = 5,
                 feature_dim: int = 128):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.extractor = FrozenBeastExtractor(checkpoint_path)

        # THE FIX: Replace frozen PCA with a trainable classical bottleneck.
        # This allows gradient descent to find the exact 4 dimensions the VQC needs.
        self.bottleneck = nn.Linear(feature_dim, n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(
            self._circuit, self.dev,
            interface="torch", diff_method="backprop",
        )

        # The quantum trainable parameters: 1 RY angle per qubit per layer
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)

        # Trainable classical readout
        self.readout = nn.Linear(n_qubits, num_classes)

    def _circuit(self, features, weights):
        """
        features : (B, n_qubits)
        weights  : (n_layers, n_qubits)
        """
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RY(features[:, q], wires=q)  # data re-upload
            for q in range(self.n_qubits):
                qml.RY(weights[layer, q], wires=q)  # trainable rotation
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def forward(self, flux, aux=None, *_):
        original_device = flux.device

        # Extract 128-d features using the frozen Beast
        feats = self.extractor(flux, aux)

        # Trainable compression down to 4 dimensions
        proj = self.bottleneck(feats)
        proj = torch.tanh(proj) * math.pi

        # Quantum Circuit
        q_list = self.qnode(proj.cpu(), self.q_weights.cpu())
        q_out = torch.stack(q_list, dim=1).float().to(original_device)

        # Final classical linear layer
        return self.readout(q_out)


# ---------------------------------------------------------------------------
# Tiny Classical head — Exactly 556 trainable params (Parameter-Matched)
# ---------------------------------------------------------------------------

class FrozenBeastTinyClassicalClassifier(nn.Module):
    """
    A purely classical network strictly matched to the Hybrid VQC's parameter count (556 params).

    Architecture structurally mirrors the Hybrid VQC:
      1. Bottleneck: Linear(128, 4) = 516 params (Matches hybrid's projection layer)
      2. Middle:     Linear(4, 4)   = 20 params  (Replaces the 20-param quantum VQC)
      3. Readout:    Linear(4, 4)   = 20 params  (Matches hybrid's classical readout)

    Total: 556 trainable parameters.
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 4, feature_dim: int = 128):
        super().__init__()
        self.extractor = FrozenBeastExtractor(checkpoint_path)

        # We use Tanh in the bottleneck to exactly mimic how the quantum circuit
        # receives bounded data, ensuring the fairest possible comparison.
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, num_classes)
        )

    def forward(self, flux, aux=None, *_):
        feats = self.extractor(flux, aux)
        return self.head(feats)


# ---------------------------------------------------------------------------
# Tiny Classical (Tanh-only) — dead-ReLU ablation, identical param count
# ---------------------------------------------------------------------------

class FrozenBeastTinyClassicalTanhClassifier(nn.Module):
    """
    Identical to FrozenBeastTinyClassicalClassifier except the inner ReLU is
    swapped for Tanh. Tests whether the ~79% cap of the param-matched classical
    head is caused by dead neurons in the width-4 ReLU layer.

    Same 556 trainable parameters, same shapes, same data flow.
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 4, feature_dim: int = 128):
        super().__init__()
        self.extractor = FrozenBeastExtractor(checkpoint_path)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, num_classes),
        )

    def forward(self, flux, aux=None, *_):
        feats = self.extractor(flux, aux)
        return self.head(feats)


# ---------------------------------------------------------------------------
# Quantum VQC head — ~550 trainable params (Hybrid Bottleneck approach)
# ---------------------------------------------------------------------------

class FrozenBeastVQCClassifier2(nn.Module):
    """
    Frozen Beast → Trainable Bottleneck (128 → n_qubits) [~516 trainable params]
        → tanh * π
        → data-reuploading VQC (n_qubits × n_layers × 1 RY = 20 trainable)
        → Linear(n_qubits, num_classes) trainable readout (20 trainable params)
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 4,
                 n_qubits: int = 4, n_layers: int = 5,
                 feature_dim: int = 128):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.extractor = FrozenBeastExtractor(checkpoint_path)

        # THE FIX: Replace frozen PCA with a trainable classical bottleneck.
        # This allows gradient descent to find the exact 4 dimensions the VQC needs.
        self.bottleneck = nn.Linear(feature_dim, n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(
            self._circuit, self.dev,
            interface="torch", diff_method="backprop",
        )

        # The quantum trainable parameters: 1 RY angle per qubit per layer
        # This now uses 4 qubits * n_layers = 60 trainable quantum params
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        # Trainable classical readout
        self.readout = nn.Linear(n_qubits, num_classes)

    def _circuit(self, features, weights):
        """
        features : (B, n_qubits)
        weights  : (n_layers, n_qubits, 3)
        """
        for layer in range(self.n_layers):
            # 1. Multi-Axis Data Encoding (0 trainable params)
            for q in range(self.n_qubits):
                # By using RX and RZ, we explore the full complex Hilbert space
                qml.RX(features[:, q], wires=q)
                qml.RZ(features[:, q], wires=q)

            # 2. Universal Trainable Rotations (SU(2))
            for q in range(self.n_qubits):
                qml.Rot(
                    weights[layer, q, 0],
                    weights[layer, q, 1],
                    weights[layer, q, 2],
                    wires=q
                )

            # 3. Alternating Entanglement (Spreads correlation faster)
            # Evens layers entangle (0,1) and (2,3)
            # Odd layers entangle (1,2) and (3,0)
            if layer % 2 == 0:
                for q in range(0, self.n_qubits, 2):
                    qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
            else:
                for q in range(1, self.n_qubits, 2):
                    qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def forward(self, flux, aux=None, *_):
        original_device = flux.device

        # Extract 128-d features using the frozen Beast
        feats = self.extractor(flux, aux)

        # Trainable compression down to 4 dimensions
        proj = self.bottleneck(feats)
        proj = torch.tanh(proj) * math.pi

        # Quantum Circuit
        q_list = self.qnode(proj.cpu(), self.q_weights.cpu())
        q_out = torch.stack(q_list, dim=1).float().to(original_device)

        # Final classical linear layer
        return self.readout(q_out)


# ---------------------------------------------------------------------------
# Frozen-PCA variants — shrink the trainable bottleneck to 0 parameters
# so the comparison isolates the head (VQC vs tiny MLP).
# ---------------------------------------------------------------------------

class FrozenBeastVQCPCAClassifier(nn.Module):
    """
    Frozen Beast → Frozen PCA(128 → n_qubits) → tanh*π → VQC → Linear readout.

    Trainable params:
      VQC weights: n_layers × n_qubits × 1 RY
      Readout:     Linear(n_qubits, num_classes)
    For n_qubits=4, n_layers=5, num_classes=4: 20 + 20 = 40 trainable params.

    Call .fit_pca(train_loader, device) once before training.
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 4,
                 n_qubits: int = 4, n_layers: int = 5,
                 feature_dim: int = 128):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.extractor = FrozenBeastExtractor(checkpoint_path)
        self.bottleneck = FrozenPCABottleneck(feature_dim, n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(
            self._circuit, self.dev,
            interface="torch", diff_method="backprop",
        )
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.readout = nn.Linear(n_qubits, num_classes)

    def _circuit(self, features, weights):
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RY(features[:, q], wires=q)
            for q in range(self.n_qubits):
                qml.RY(weights[layer, q], wires=q)
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def fit_pca(self, loader, device, max_batches=None):
        _fit_pca_on_loader(self.extractor, self.bottleneck, loader, device, max_batches)

    def forward(self, flux, aux=None, *_):
        original_device = flux.device
        feats = self.extractor(flux, aux)
        proj = self.bottleneck(feats)
        proj = torch.tanh(proj) * math.pi
        q_list = self.qnode(proj.cpu(), self.q_weights.cpu())
        q_out = torch.stack(q_list, dim=1).float().to(original_device)
        return self.readout(q_out)


class FrozenBeastVQCHybridClassifier(nn.Module):
    """
    Hybrid PCA + small trainable Linear bottleneck.

    Frozen Beast → Frozen PCA(128 → pca_dim) → Linear(pca_dim → n_qubits)
                 → tanh·π → VQC → Linear(n_qubits, num_classes)

    The 128→n_qubits projection is still learnable end-to-end, but factored
    into a frozen variance-maximising PCA followed by a tiny trainable Linear.
    Same expressive shape as V3's single Linear(128, 4), just low-rank-constrained.

    Trainable params:
      Linear(pca_dim, n_qubits): pca_dim·n_qubits + n_qubits
      VQC weights:               n_layers · n_qubits
      Readout:                   n_qubits·num_classes + num_classes

    For pca_dim=16, n_qubits=4, n_layers=5, num_classes=4:
      Linear(16, 4): 68
      VQC:           20
      Readout:       20
      Total:        108 params  (vs 556 in V3 → 5× smaller)

    Call .fit_pca(train_loader, device) once before training.
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 4,
                 n_qubits: int = 4, n_layers: int = 5,
                 pca_dim: int = 16, feature_dim: int = 128):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.pca_dim = pca_dim

        self.extractor = FrozenBeastExtractor(checkpoint_path)
        self.pca = FrozenPCABottleneck(feature_dim, pca_dim)
        self.bottleneck = nn.Linear(pca_dim, n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(
            self._circuit, self.dev,
            interface="torch", diff_method="backprop",
        )
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.readout = nn.Linear(n_qubits, num_classes)

    def _circuit(self, features, weights):
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RY(features[:, q], wires=q)
            for q in range(self.n_qubits):
                qml.RY(weights[layer, q], wires=q)
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def fit_pca(self, loader, device, max_batches=None):
        _fit_pca_on_loader(self.extractor, self.pca, loader, device, max_batches)

    def forward(self, flux, aux=None, *_):
        original_device = flux.device
        feats = self.extractor(flux, aux)
        z = self.pca(feats)
        z = self.bottleneck(z)
        z = torch.tanh(z) * math.pi
        q_list = self.qnode(z.cpu(), self.q_weights.cpu())
        q_out = torch.stack(q_list, dim=1).float().to(original_device)
        return self.readout(q_out)


class FrozenBeastTinyClassicalPCAClassifier(nn.Module):
    """
    Frozen Beast → Frozen PCA(128 → n_features) → tanh*π → tiny MLP head.

    Head shape mirrors the quantum head:
      Linear(n_features, n_features)   ← mirrors VQC mid-block
      Tanh                              ← matches VQC's bounded activation
      Linear(n_features, num_classes)   ← mirrors readout

    For n_features=4, num_classes=4: 20 + 20 = 40 trainable params,
    matching FrozenBeastVQCPCAClassifier exactly.

    Call .fit_pca(train_loader, device) once before training.
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 4,
                 n_features: int = 4, feature_dim: int = 128):
        super().__init__()
        self.extractor = FrozenBeastExtractor(checkpoint_path)
        self.bottleneck = FrozenPCABottleneck(feature_dim, n_features)
        self.head = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Tanh(),
            nn.Linear(n_features, num_classes),
        )

    def fit_pca(self, loader, device, max_batches=None):
        _fit_pca_on_loader(self.extractor, self.bottleneck, loader, device, max_batches)

    def forward(self, flux, aux=None, *_):
        feats = self.extractor(flux, aux)
        proj = self.bottleneck(feats)
        proj = torch.tanh(proj) * math.pi
        return self.head(proj)