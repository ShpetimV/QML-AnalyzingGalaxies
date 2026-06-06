import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA

from src.param_config import SDSSDataConfig
from src.sdss_dataloader import SDSSDataModule
from src.models.classical_cnn import SpectraClassifier

# --- CONFIGURATION ---
TARGET_CLASSES = ["GALAXY_STARBURST", "GALAXY_STARFORMING", "QSO_STARBURST_BROADLINE", "QSO_STARFORMING"] # , "GALAXY_AGN", "QSO_AGN"]
NUM_CLASSES = len(TARGET_CLASSES)
PARQUET_PATH = PROJECT_ROOT / "dataset" / "ML_SDSS_CLEANED_DATA.parquet"
MODEL_WEIGHTS = "../../src/models/trained_models/baseline_cnn_transformer.pt"
OUTPUT_DIR = f"./data/data_fewshot_{NUM_CLASSES}_classes"
N_COMPONENTS = 16
SAMPLE_SIZES = [25, 50, 100, 150, 200, 250, 300, 350, 400, 500]


def extract_features(model, loader, device):
    """Passes data through the frozen model and collects the output vectors."""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            flux = batch['flux'].to(device)
            labels = batch['label'].numpy()

            # The model now outputs the 1024D GAP features (replaced the classifier)
            features = model(flux, aux=None)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels)

    return np.vstack(all_features), np.array(all_labels)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 1. Load the Data (Strictly NO augmentation for few-shot evaluation)
    print(f"\nLoading data for classes: {TARGET_CLASSES}...")
    data_config = SDSSDataConfig(
        parquet_path=str(PARQUET_PATH),
        use_augmentation=False,  # No augmentation for few-shot feature extraction
        num_workers=0)

    data_module = SDSSDataModule(data_config)
    data_module.prepare_data(classes=TARGET_CLASSES)

    train_loader = data_module.get_loader(data_module.train_ds, use_sampler=False)
    test_loader = data_module.get_loader(data_module.test_ds, use_sampler=False)

    # 2. Load the "Frozen Beast"
    print("\nLoading the 62-class Frozen Beast...")
    # MUST initialize with 62 classes to match the saved weights
    model = SpectraClassifier(num_classes=62, aux_features=0, dropout=0.0)

    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: Could not find weights at {MODEL_WEIGHTS}. Please update the path.")
        return

    # Chop off the head and freeze
    model.classifier = nn.Identity()
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)

    # 3. Extract 1024D Features
    print("\nExtracting features through the CNN + Transformer...")
    x_train_full, y_train_full = extract_features(model, train_loader, device)
    x_test_full, y_test_full = extract_features(model, test_loader, device)

    print(f"Extracted Train Shape: {x_train_full.shape}")
    print(f"Extracted Test Shape:  {x_test_full.shape}")

    # 4. Fit PCA to the maximum training set size we intend to use
    print(f"\nFitting PCA to reduce from {x_train_full.shape[1]}D to {N_COMPONENTS}D...")

    max_samples = max(SAMPLE_SIZES)
    idx_pca_fit_list = []

    # Dynamically grab max_samples for EVERY class
    for c in range(NUM_CLASSES):
        idx_c = np.nonzero(y_train_full == c)[0]
        idx_pca_fit_list.append(idx_c[:max_samples])

    idx_pca_fit = np.concatenate(idx_pca_fit_list)

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(x_train_full[idx_pca_fit])

    # Transform everything
    x_train_pca = pca.transform(x_train_full)
    x_test_pca = pca.transform(x_test_full)

    # 5. Create and Save the Few-Shot Subsets
    print("\nGenerating Few-Shot datasets...")
    for n in SAMPLE_SIZES:

        # Dynamically grab 'n' samples for EVERY class
        subset_idx_list = []
        for c in range(NUM_CLASSES):
            idx_c = np.nonzero(y_train_full == c)[0]
            subset_idx_list.append(idx_c[:n])

        subset_idx = np.concatenate(subset_idx_list)

        x_train_n = x_train_pca[subset_idx]
        y_train_n = y_train_full[subset_idx]

        # Shuffle
        generator = np.random.default_rng(seed=42 + n)
        shuffle_idx = generator.permutation(len(subset_idx))
        x_train_n = x_train_n[shuffle_idx]
        y_train_n = y_train_n[shuffle_idx]

        np.save(os.path.join(OUTPUT_DIR, f"X_train_{n}.npy"), x_train_n)
        np.save(os.path.join(OUTPUT_DIR, f"y_train_{n}.npy"), y_train_n)
        print(f" Saved {n}-shot training set: {x_train_n.shape}")

    # same test set for all sizes -> fair evaluation
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), x_test_pca)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test_full)
    print(f" Saved universal Test set: {x_test_pca.shape}")

    print("\nFeature extraction complete! Ready for QML training.")


if __name__ == "__main__":
    main()
