import os
import torch
import pennylane as qml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import your model
from src.models.quantum_model import AngleEncodingClassifier


def trace_cnn_shapes(model):
    """Passes a dummy tensor through the CNN and prints the shape at every step."""
    print("=" * 50)
    print("🔍 CLASSICAL CNN SHAPE TRACE")
    print("=" * 50)

    # Create a dummy batch of 2 spectra, 1 channel, 4448 bins
    x = torch.randn(2, 1, 4448)
    print(f"Input Flux Shape:   {list(x.shape)}")
    print("-" * 50)

    # Trace through the backbone
    for i, layer in enumerate(model.extractor.backbone):
        x = layer(x)
        layer_name = layer.__class__.__name__
        # Just to make the printout look clean and aligned
        print(f"After {layer_name:<16}: {list(x.shape)}")

    # Flatten
    x = x.reshape(x.size(0), -1)
    print("-" * 50)
    print(f"After Flatten       : {list(x.shape)}")
    print("-" * 50)

    # Trace through the MLP projections
    for i, layer in enumerate(model.extractor.proj):
        x = layer(x)
        layer_name = layer.__class__.__name__
        print(f"After {layer_name:<16}: {list(x.shape)}")

    print("=" * 50)
    print(f"🎯 Final Extracted Features: {list(x.shape)}")
    print("=" * 50)


def draw_quantum_circuit(model, save_path="quantum_circuit.png"):
    """Uses PennyLane to draw the VQC and saves it as an image."""
    print("\n🎨 Generating Quantum Circuit Image...")

    # Create dummy inputs for the quantum node
    # Batch size of 1, n_qubits=4
    dummy_features = torch.rand(1, model.n_qubits)
    dummy_weights = model.q_weights.detach()

    # Create the drawing figure using PennyLane's Matplotlib drawer
    fig, ax = qml.draw_mpl(model.qnode, decimals=2)(dummy_features, dummy_weights)

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Quantum circuit saved successfully to: ./{save_path}")


if __name__ == "__main__":
    # Initialize the model with your exact settings
    model = AngleEncodingClassifier(
        num_classes=2,
        n_qubits=4,
        n_layers=6,
    )

    # 1. Print the Classical CNN Trace
    trace_cnn_shapes(model)

    # 2. Draw and Save the Quantum Circuit
    draw_quantum_circuit(model, "quantum_circuit.png")