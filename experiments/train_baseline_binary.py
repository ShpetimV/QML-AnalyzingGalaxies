import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.param_config import SDSSDataConfig, TrainingConfig
from src.sdss_dataloader import SDSSDataModule
from src.models.classical_cnn import SpectraClassifier
from src.training.trainer import SDSSPerformanceTrainer
from src.training.metrics import SDSSMetricTracker

# ---------------------------------------------------------------------------
# Binary task config — same classes as quantum experiment
# ---------------------------------------------------------------------------
CLASS_A = "STAR_BROWN_DWARF_L"      # → label 0
CLASS_B = "STAR_M8"   # → label 1


# ---------------------------------------------------------------------------
# Binary dataset wrapper (identical to quantum version)
# ---------------------------------------------------------------------------
class BinarySubset(Dataset):
    """Filters an SDSSDataset to two classes and relabels them 0/1."""

    def __init__(self, base_dataset, class_a_idx, class_b_idx):
        mask = (base_dataset.labels == class_a_idx) | (base_dataset.labels == class_b_idx)
        self.indices = torch.where(mask)[0]
        self.base = base_dataset
        self.class_a_idx = class_a_idx
        self.class_b_idx = class_b_idx

        old_labels = base_dataset.labels[self.indices]
        self.labels = (old_labels == class_b_idx).long()
        self.is_train = base_dataset.is_train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.base[self.indices[idx].item()]
        sample['label'] = self.labels[idx]
        return sample


# ---------------------------------------------------------------------------
# Extra binary metrics (ROC + PR curves)
# ---------------------------------------------------------------------------
def save_roc_pr_curves(y_true, y_probs, results_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax1.set_xlabel('False positive rate'); ax1.set_ylabel('True positive rate')
    ax1.set_title('ROC curve'); ax1.legend(); ax1.grid(True, alpha=0.3)

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    ax2.plot(recall, precision, 'r-', linewidth=2, label=f'AP = {ap:.4f}')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('Precision-recall curve'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "roc_pr_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved ROC/PR curves → {path}")
    return auc, ap


def main():
    # 1. Setup Configuration & Data
    data_config = SDSSDataConfig()
    training_config = TrainingConfig()
    results_dir = "results_classical_binary"
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
                              sampler=sampler, num_workers=data_config.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=data_config.batch_size,
                              shuffle=False, num_workers=data_config.num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=data_config.batch_size,
                              shuffle=False, num_workers=data_config.num_workers,
                              pin_memory=True)

    # 3. Initialize Classical CNN (2 classes)
    model = SpectraClassifier(
        num_classes=2,
        aux_features=len(data_config.scalar_cols),
        dropout=training_config.dropout
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nClassical CNN: {total_params:,} params")

    # 4. Initialize Trainer & Metrics
    trainer = SDSSPerformanceTrainer(model, training_config)
    tracker = SDSSMetricTracker(results_dir=results_dir)

    # 5. Run Training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config.epochs,
        lr=training_config.lr,
        weight_decay=training_config.weight_decay
    )

    # 6. Final Evaluation
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load("best_baseline_model.pt", map_location=trainer.device))
    model.to(trainer.device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            flux = batch['flux'].to(trainer.device)
            aux = batch['scalars'].to(trainer.device)
            labels = batch['label'].to(trainer.device)

            logits = model(flux, aux)
            probs = torch.softmax(logits, dim=1)

            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # 7. Generate Insights
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    tracker.plot_history(trainer.history)
    tracker.plot_confusion_matrix(y_true, y_pred, binary_classes)
    tracker.plot_per_class_accuracy(y_true, y_pred, binary_classes)

    # Binary-specific: ROC and Precision-Recall curves
    auc, ap = save_roc_pr_curves(y_true, y_probs, results_dir)

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=binary_classes, digits=4))
    print(f"ROC AUC: {auc:.4f}  |  Avg Precision: {ap:.4f}")
    print(f"\nClassical Binary Complete! Results saved to ./{results_dir}")


if __name__ == "__main__":
    main()