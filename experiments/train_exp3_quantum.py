import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import torch
import numpy as np
from sklearn.metrics import classification_report

from src.param_config import SDSSDataConfig, TrainingConfig
from src.sdss_dataloader import SDSSDataModule
from src.training.trainer import SDSSPerformanceTrainer
from src.training.metrics import SDSSMetricTracker
from src.models.exp3_models import FrozenBeastVQCClassifier

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Experiment 3 — multi-class hard task (4 classes drawn from confusion pairs)
# ---------------------------------------------------------------------------
CLASSES = [
    "STAR_BROWN_DWARF_L",
    "STAR_M8",
    "GALAXY_STARBURST",
    "GALAXY_STARFORMING",
]
SAMPLES_PER_CLASS = 1000

N_QUBITS = 4
N_LAYERS = 5

BEAST_CHECKPOINT = "src/models/trained_models/baseline_cnn_transformer.pt"


def subsample_balanced(dataset, n_per_class, num_classes, seed=42):
    """Subsample dataset.indices in-place so each class has at most n_per_class samples."""
    rng = np.random.default_rng(seed)
    current_labels = dataset.full_labels[dataset.indices]
    new_indices = []
    for class_idx in range(num_classes):
        class_positions = np.where(current_labels == class_idx)[0]
        if len(class_positions) > n_per_class:
            chosen = rng.choice(class_positions, n_per_class, replace=False)
        else:
            chosen = class_positions
        new_indices.append(dataset.indices[chosen])
    dataset.indices = np.concatenate(new_indices)




def main():
    DO_TRAINING = True

    data_config = SDSSDataConfig(num_workers=0)
    training_config = TrainingConfig()
    results_dir = f"results_exp3_quantum_{len(CLASSES)}class"
    os.makedirs(results_dir, exist_ok=True)

    data_module = SDSSDataModule(data_config)
    data_module.prepare_data(classes=CLASSES)

    num_classes = len(CLASSES)

    # Subsample to ~1000/class total, distributed proportionally over the splits
    subsample_balanced(data_module.train_ds, int(SAMPLES_PER_CLASS * data_config.train_size), num_classes)
    subsample_balanced(data_module.val_ds, int(SAMPLES_PER_CLASS * data_config.val_size), num_classes)
    subsample_balanced(data_module.test_ds, int(SAMPLES_PER_CLASS * data_config.test_size), num_classes)

    print(f"\n{len(CLASSES)}-class task: {CLASSES}")
    for name, ds in [("Train", data_module.train_ds), ("Val", data_module.val_ds), ("Test", data_module.test_ds)]:
        labels = ds.labels
        counts = {c: (labels == i).sum().item() for i, c in enumerate(CLASSES)}
        print(f"  {name} ({len(ds)}): {counts}")

    train_loader = data_module.get_loader(data_module.train_ds, use_sampler=True)
    val_loader = data_module.get_loader(data_module.val_ds)
    test_loader = data_module.get_loader(data_module.test_ds)

    # Quantum head with frozen Beast extractor
    model = FrozenBeastVQCClassifier(
        checkpoint_path=BEAST_CHECKPOINT,
        num_classes=num_classes,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"\nFrozenBeast + VQC head")
    print(f"  Trainable params: {trainable}  ← should be ~556 (Bottleneck + VQC + Readout)")
    print(f"  Frozen params:    {frozen:,}  (Beast extractor + PCA projection buffer)")

    trainer = SDSSPerformanceTrainer(model, training_config, run_name="Exp3Quantum_V3")
    tracker = SDSSMetricTracker(results_dir=results_dir)

    if DO_TRAINING:
        print("\n--- Fitting PCA Matrix ---")
        # Run fit_pca using our adapter before training begins

        print("\n--- Starting Training ---")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config.epochs,
            lr=training_config.lr,
            weight_decay=training_config.weight_decay,
        )
    else:
        print("\n--- Skipping Training (Evaluation Only Mode) ---")

    # Load best checkpoint
    print("\n--- Evaluating on Test Set ---")
    list_of_runs = glob.glob("runs/Exp3Quantum_*")
    if list_of_runs:
        latest_run = max(list_of_runs, key=os.path.getctime)
        best_model_path = os.path.join(latest_run, "trained_models", "best_model.pt")
        print(f"Loading weights from: {best_model_path}")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=trainer.device))
            # Manually flag PCA as fitted since boolean flags aren't saved in state_dicts
            model.pca_fitted = True
        except FileNotFoundError:
            print("Could not find best_model.pt. Using current weights.")

    model.to(trainer.device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            flux = batch['flux'].to(trainer.device)
            labels = batch['label'].to(trainer.device)
            logits = model(flux)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    tracker.plot_history(trainer.history)
    tracker.plot_confusion_matrix(y_true, y_pred, CLASSES)
    tracker.plot_per_class_accuracy(y_true, y_pred, CLASSES)

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
    print(f"\n[Exp3 Quantum] Trainable params: {trainable}")
    print(f"Results saved to ./{results_dir}")


if __name__ == "__main__":
    main()