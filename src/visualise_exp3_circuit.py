"""
Render the quantum circuit used in Experiment 3 (FrozenBeastVQCClassifier).

Per layer (5 layers total, 4 qubits):
  1. RY data re-upload — angle = projected feature value (from trainable bottleneck)
  2. RY trainable rotation — one trainable angle per qubit
  3. Circular CNOT entanglement — q → (q+1) % n_qubits

Output: PauliZ expectation values on every qubit (4 logits, then read out by Linear(4,4)).

Usage (from project root):
    uv run python src/visualise_exp3_circuit.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
import torch

N_QUBITS = 4
N_LAYERS = 5
SAVE_PATH = "exp3_quantum_circuit.png"


def exp3_circuit(features, weights, n_qubits=N_QUBITS, n_layers=N_LAYERS):
    """Mirrors FrozenBeastVQCClassifier._circuit exactly."""
    for layer in range(n_layers):
        for q in range(n_qubits):
            qml.RY(features[q], wires=q)            # data re-upload
        for q in range(n_qubits):
            qml.RY(weights[layer, q], wires=q)      # trainable rotation
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]


def main():
    dev = qml.device("default.qubit", wires=N_QUBITS)
    qnode = qml.QNode(exp3_circuit, dev)

    # Dummy inputs just for drawing — values don't matter for the diagram
    features = torch.linspace(-1.0, 1.0, N_QUBITS)
    weights = torch.randn(N_LAYERS, N_QUBITS) * 0.1

    fig, _ = qml.draw_mpl(qnode, decimals=2)(features, weights)
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved circuit diagram → ./{SAVE_PATH}")
    print(f"  Qubits: {N_QUBITS}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Trainable VQC params: {N_QUBITS * N_LAYERS}  (1 RY per qubit per layer)")


if __name__ == "__main__":
    main()
