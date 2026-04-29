import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import classification_report

from src.param_config import SDSSDataConfig, TrainingConfig
from src.sdss_dataloader import SDSSDataModule
from src.models.classical_cnn import SpectraClassifier
from src.training.trainer import SDSSPerformanceTrainer
from src.training.metrics import SDSSMetricTracker

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Binary task config — same classes as quantum experiment
# ---------------------------------------------------------------------------
CLASS_A = "STAR_BROWN_DWARF_L"  # → label 0
CLASS_B = "STAR_M8"             # → label 1


def main():
    DO_TRAINING = True

    # 1. Setup Configuration & Data
    data_config = SDSSDataConfig()
    training_config = TrainingConfig()
    results_dir = "results_classical_binary"
    os.makedirs(results_dir, exist_ok=True)

    data_module = SDSSDataModule(data_config)
    data_module.prepare_data(classes=[CLASS_A, CLASS_B])

    binary_classes = list(data_module.classes)
    print(f"Binary task: {CLASS_A} (0) vs {CLASS_B} (1)")
    for name, ds in [("Train", data_module.train_ds), ("Val", data_module.val_ds), ("Test", data_module.test_ds)]:
        labels = ds.labels
        print(f"  {name}: {CLASS_A}={(labels==0).sum().item()}  {CLASS_B}={(labels==1).sum().item()}")

    train_loader = data_module.get_loader(data_module.train_ds, use_sampler=True)
    val_loader   = data_module.get_loader(data_module.val_ds)
    test_loader  = data_module.get_loader(data_module.test_ds)

    # 2. Initialize Classical CNN (2 classes, no auxiliary features)
    model = SpectraClassifier(
        num_classes=2,
        aux_features=0,
        dropout=training_config.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nClassical CNN: {total_params:,} params")

    # 3. Initialize Trainer & Metrics
    trainer = SDSSPerformanceTrainer(model, training_config, run_name="Baseline_CNN")
    tracker = SDSSMetricTracker(results_dir=results_dir)

    # 4. Run Training
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

    # 5. Load best checkpoint
    print("\n--- Evaluating on Test Set ---")
    model_path = "runs/Baseline_CNN_20260416_140137/trained_models/best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=trainer.device))
    model.to(trainer.device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            flux   = batch['flux'].to(trainer.device)
            labels = batch['label'].to(trainer.device)
            logits = model(flux, None)
            probs  = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # 6. Generate Metrics & Plots
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
    print(f"\nClassical Binary Complete! Results saved to ./{results_dir}")


if __name__ == "__main__":
    main()
