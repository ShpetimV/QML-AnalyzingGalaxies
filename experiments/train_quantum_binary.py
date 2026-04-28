import sys, os
from xmlrpc.client import DateTime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report

from src.param_config import SDSSDataConfig, TrainingConfig
from src.sdss_dataloader import SDSSDataModule, BinarySubset
from src.training.trainer import SDSSPerformanceTrainer
from src.training.metrics import SDSSMetricTracker
from src.models.quantum_model import AngleEncodingClassifier

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Binary task config
# ---------------------------------------------------------------------------
CLASS_A = "STAR_BROWN_DWARF_L"  # → label 0
CLASS_B = "STAR_M8"             # → label 1

N_QUBITS = 4
N_LAYERS = 6


def main():
    DO_TRAINING = True

    # 1. Setup Configuration & Data
    data_config = SDSSDataConfig(num_workers=0)  # 0 workers safer for quantum
    training_config = TrainingConfig()
    # Make Directory with todays date for results to avoid overwriting previous runs
    results_dir = f"results_quantum_binary_{CLASS_A}_vs_{CLASS_B}"
    os.makedirs(results_dir, exist_ok=True)

    data_module = SDSSDataModule(data_config)
    data_module.prepare_data()

    # 2. Filter to binary classes
    all_classes = list(data_module.classes)
    idx_a = all_classes.index(CLASS_A)
    idx_b = all_classes.index(CLASS_B)
    binary_classes = [CLASS_A, CLASS_B]

    print(f"Binary task: {CLASS_A} (0) vs {CLASS_B} (1)")

    train_ds = BinarySubset(data_module.train_ds, idx_a, idx_b)
    val_ds   = BinarySubset(data_module.val_ds,   idx_a, idx_b)
    test_ds  = BinarySubset(data_module.test_ds,  idx_a, idx_b)

    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        n0 = (ds.labels == 0).sum().item()
        n1 = (ds.labels == 1).sum().item()
        print(f"  {name}: {CLASS_A}={n0}  {CLASS_B}={n1}")

    # Balanced sampler for training
    train_labels = train_ds.labels.numpy()
    class_counts = np.bincount(train_labels, minlength=2)
    sample_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=data_config.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=data_config.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=data_config.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # 3. Initialize Quantum Model
    model = AngleEncodingClassifier(
        num_classes=2,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        dropout=training_config.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nQuantum model: {N_QUBITS} qubits, {N_LAYERS} layers")
    print(f"  Total params: {total_params:,}  (quantum: {model.q_weights.numel()}, classical: {total_params - model.q_weights.numel()})")

    # 4. Initialize Trainer & Metrics
    trainer = SDSSPerformanceTrainer(model, training_config, run_name="QuantumBinary")
    tracker = SDSSMetricTracker(results_dir=results_dir)

    # 5. Run Training
    if DO_TRAINING:
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

    # 6. Load best checkpoint
    print("\n--- Evaluating on Test Set ---")

    list_of_runs = glob.glob("runs/QuantumBinary_*")
    if list_of_runs:
        latest_run = max(list_of_runs, key=os.path.getctime)
        best_model_path = os.path.join(latest_run, "trained_models", "best_model.pt")
        print(f"Loading weights from: {best_model_path}")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=trainer.device))
        except FileNotFoundError:
            print("Could not find best_model.pt. Using current weights.")
    else:
        print("No run folders found. Evaluating with current weights.")

    model.to(trainer.device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            flux   = batch['flux'].to(trainer.device)
            labels = batch['label'].to(trainer.device)
            logits = model(flux)
            probs  = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # 7. Generate Metrics & Plots
    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    tracker.plot_history(trainer.history)
    tracker.plot_confusion_matrix(y_true, y_pred, binary_classes)
    tracker.plot_per_class_accuracy(y_true, y_pred, binary_classes)
    auc, ap = tracker.plot_roc_pr_curves(y_true, y_probs)

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=binary_classes, digits=4))
    print(f"ROC AUC: {auc:.4f}  |  Avg Precision: {ap:.4f}")
    print(f"\nResults saved to ./{results_dir}")


if __name__ == "__main__":
    main()
