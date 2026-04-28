import torch
from torchview import draw_graph

# Import just the CNN part of your model
from src.models.quantum_model import SpectralFeatureExtractor


def main():
    print("Initializing Spectral Feature Extractor...")
    # Initialize the extractor
    extractor = SpectralFeatureExtractor(out_features=4)

    # Define the exact input shape your dataloader uses
    # (Batch Size=2, Channels=1, Length=4448)
    input_shape = (2, 1, 4448)

    print("Tracing the computational graph and rendering image...")

    # draw_graph traces the tensor sizes through every single layer
    model_graph = draw_graph(
        extractor,
        input_size=input_shape,
        graph_name="SpectralFeatureExtractor",
        save_graph=True,
        filename="cnn_architecture",
        expand_nested=True,  # Critical: This unrolls nn.Sequential blocks so you see inside them
        roll=True,  # Groups repeated identical layers to save space
        hide_module_functions=False,
    )

    print("✅ Success! Check your folder for 'cnn_architecture.png'")


if __name__ == "__main__":
    main()