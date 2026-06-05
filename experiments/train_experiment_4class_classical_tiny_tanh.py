"""
EXPERIMENT 2 -- Model D: param-matched classical control, Tanh (DECISIVE control).

Frozen Beast -> Linear(128,4) -> Tanh -> Linear(4,4) -> Tanh -> Linear(4,4).
Identical to Model C except the inner ReLU is swapped for Tanh. 556 trainable
params. Recovers to ~96% / 0.94 brown-dwarf recall, matching the quantum head ->
proves the C failure was a dead-ReLU artifact, so Experiment 3 is parity, not a
quantum advantage. This script supports --epochs and dumps history.json.
Output: results_exp3b_tiny_classical_tanh_4class/ (or --out-dir).
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
from src.models.exp3_models import FrozenBeastTinyClassicalTanhClassifier

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Exp2— Tiny Classical (Tanh-only): ablates the inner ReLU from the
# parameter-matched classical mirror to test the dead-ReLU hypothesis.
# Architecture: Linear(128,4) -> Tanh -> Linear(4,4) -> Tanh -> Linear(4,K)
# ---------------------------------------------------------------------------
CLASSES = [
    "STAR_BROWN_DWARF_L",
    "STAR_M8",
    "GALAXY_STARBURST",
    "GALAXY_STARFORMING",
]
SAMPLES_PER_CLASS = 1000
BEAST_CHECKPOINT = "src/models/trained_models/baseline_cnn_transformer.pt"
RUN_NAME = "Exp3bTinyClassicalTanh"


def subsample_balanced(dataset, n_per_class, num_classes, seed=42):
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

    results_dir = f"results_exp3b_tiny_classical_tanh_{len(CLASSES)}class"
    os.makedirs(results_dir, exist_ok=True)

    data_module = SDSSDataModule(data_config)
    data_module.prepare_data(classes=CLASSES)

    num_classes = len(CLASSES)

    subsample_balanced(data_module.train_ds, int(SAMPLES_PER_CLASS * data_config.train_size), num_classes)
    subsample_balanced(data_module.val_ds,   int(SAMPLES_PER_CLASS * data_config.val_size),   num_classes)
    subsample_balanced(data_module.test_ds,  int(SAMPLES_PER_CLASS * data_config.test_size),  num_classes)

    print(f"\n{len(CLASSES)}-class task: {CLASSES}")
    for name, ds in [("Train", data_module.train_ds), ("Val", data_module.val_ds), ("Test", data_module.test_ds)]:
        labels = ds.labels
        counts = {c: (labels == i).sum().item() for i, c in enumerate(CLASSES)}
        print(f"  {name} ({len(ds)}): {counts}")

    train_loader = data_module.get_loader(data_module.train_ds, use_sampler=True)
    val_loader   = data_module.get_loader(data_module.val_ds)
    test_loader  = data_module.get_loader(data_module.test_ds)

    model = FrozenBeastTinyClassicalTanhClassifier(
        checkpoint_path=BEAST_CHECKPOINT,
        num_classes=num_classes,
        feature_dim=128,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"\nFrozenBeast + Tiny Classical Head (Tanh-only)")
    print(f"  Trainable params: {trainable}  ← should match VQC (556)")
    print(f"  Frozen params:    {frozen:,}  (Beast extractor)")

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

    model.to(trainer.device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            flux   = batch['flux'].to(trainer.device)
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
    print(f"\n[Exp3b Tiny Classical Tanh] Trainable params: {trainable}")
    print(f"Results saved to ./{results_dir}")


if __name__ == "__main__":
    main()
