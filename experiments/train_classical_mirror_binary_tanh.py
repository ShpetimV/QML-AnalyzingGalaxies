"""
EXPERIMENT 1 (binary) -- param-matched classical control, Tanh (dead-ReLU fix).

This script trains a binary classifier using a "Classical Mirror" architecture with Tanh activation, on the SDSS dataset.

This experiment was conducted due to the observation that the original ClassicalMirrorClassifier with ReLU activations
had a significant portion of its parameters (the "mirror layer") effectively unused due to dead ReLUs.
By replacing ReLU with Tanh, we can keep all parameters active and see if this leads to improved performance
on the same binary classification task (STAR_BROWN_DWARF_L vs STAR_M8).
"""

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
from src.models.classical_mirror import ClassicalMirrorTanhClassifier

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Binary task — dead-ReLU ablation of ClassicalMirrorClassifier.
# Architecture diff vs original:
#   Linear(4,8) -> ReLU -> Linear(8,4)   (original, 76 params, ~46% collapse)
#   Linear(4,8) -> Tanh -> Linear(8,4)   (this script, same 76 params)
# Everything else (extractor, head, data, hyperparameters) is identical.
# ---------------------------------------------------------------------------
CLASS_A = "STAR_BROWN_DWARF_L"  # → label 0
CLASS_B = "STAR_M8"             # → label 1

N_FEATURES = 8
RUN_NAME = f"ClassicalMirrorTanh_{N_FEATURES}features"


def main():
    DO_TRAINING = True

    data_config = SDSSDataConfig(num_workers=0)
    training_config = TrainingConfig()
    results_dir = f"results_classical_mirror_tanh_binary_{CLASS_A}_vs_{CLASS_B}_{N_FEATURES}features"
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

    model = ClassicalMirrorTanhClassifier(
        num_classes=2,
        n_features=N_FEATURES,
        dropout=training_config.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mirror_params = sum(p.numel() for p in model.classical_layer.parameters())
    print(f"\nClassical Mirror (Tanh) model: {N_FEATURES} features")
    print(f"  Total params: {total_params:,}  (mirror layer: {mirror_params}, rest: {total_params - mirror_params})")

    trainer = SDSSPerformanceTrainer(model, training_config, run_name=RUN_NAME)
    tracker = SDSSMetricTracker(results_dir=results_dir)

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

    print("\n--- Evaluating on Test Set ---")

    list_of_runs = glob.glob(f"runs/{RUN_NAME}_*")
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
