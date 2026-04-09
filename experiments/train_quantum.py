"""
Training Script for Quantum SDSS Spectral Classifiers
======================================================
Uses the existing SDSSDataModule with augmentation + cropping,
reading from ML_SDSS_CLEANED_DATA.parquet.

Supports both angle and amplitude encoding via CLI args.

Usage (from project root):
    uv run python models/train_quantum.py --encoding angle
    uv run python models/train_quantum.py --encoding amplitude
    uv run python models/train_quantum.py --encoding angle --n_qubits 10 --n_layers 6
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Path setup: add project root AND models/ to sys.path ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from src.param_config import SDSSDataConfig
from src.sdss_dataloader import SDSSDataModule
from src.models.quantum_model import get_quantum_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints_quantum")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "models", "results_quantum")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# Device — PennyLane default.qubit uses CPU; MPS/CUDA for classical parts
DEVICE = "cpu"   # quantum simulation runs on CPU anyway


# ---------------------------------------------------------------------------
# Training + Evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, log_interval=20):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, batch in enumerate(loader):
        flux    = batch['flux'].to(device)       # [B, 1, L]
        scalars = batch['scalars'].to(device)    # [B, 6]
        labels  = batch['label'].to(device)      # [B]

        optimizer.zero_grad()
        logits = model(flux, scalars)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}  "
                  f"acc={correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        flux    = batch['flux'].to(device)
        scalars = batch['scalars'].to(device)
        labels  = batch['label'].to(device)

        logits = model(flux, scalars)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_training_curves(history, encoding_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss — {encoding_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy — {encoding_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"training_curves_{encoding_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved training curves → {path}")


def save_confusion_matrix(y_true, y_pred, class_names, encoding_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {encoding_name}')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1%})',
                    ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_matrix_{encoding_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Quantum SDSS Classifier")
    parser.add_argument("--encoding", type=str, default="angle",
                        choices=["angle", "amplitude"],
                        help="Quantum encoding strategy")
    parser.add_argument("--n_qubits", type=int, default=8,
                        help="Number of qubits (default: 8)")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of variational layers (default: 4)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64, broadcasting makes larger batches efficient)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--subset", type=int, default=0,
                        help="Use only N samples for fast experimentation (0 = full dataset)")
    args = parser.parse_args()

    encoding_name = f"{args.encoding}_{args.n_qubits}q_{args.n_layers}L"
    print(f"\n{'='*60}")
    print(f"  Quantum SDSS Classifier — {encoding_name}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Data — use existing SDSSDataModule with augmentation
    # ------------------------------------------------------------------
    print("[1/5] Loading data via SDSSDataModule...")
    config = SDSSDataConfig(
        parquet_path=os.path.join(PROJECT_ROOT, 'dataset', 'ML_SDSS_CLEANED_DATA.parquet'),
        batch_size=args.batch_size,
        num_workers=0,         # safer for quantum (avoids PennyLane fork issues)
    )
    dm = SDSSDataModule(config)
    dm.prepare_data()

    # Optional: use a subset for fast experimentation
    if args.subset > 0:
        from torch.utils.data import Subset
        def _subset(ds, n):
            n = min(n, len(ds))
            indices = torch.randperm(len(ds))[:n].tolist()
            sub = Subset(ds, indices)
            # Carry over attributes the loader/sampler needs
            sub.labels = ds.labels[indices]
            sub.is_train = ds.is_train
            return sub

        dm.train_ds = _subset(dm.train_ds, args.subset)
        dm.val_ds   = _subset(dm.val_ds, args.subset // 4)
        dm.test_ds  = _subset(dm.test_ds, args.subset // 4)
        print(f"  ⚡ Subset mode: train={len(dm.train_ds)} val={len(dm.val_ds)} test={len(dm.test_ds)}")

    train_loader = dm.get_loader(dm.train_ds, use_sampler=True)
    val_loader   = dm.get_loader(dm.val_ds)
    test_loader  = dm.get_loader(dm.test_ds)

    num_classes = dm.num_classes
    class_names = list(dm.classes)
    n_scalars   = len(config.scalar_cols)

    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Train: {len(dm.train_ds)}  Val: {len(dm.val_ds)}  Test: {len(dm.test_ds)}")
    print(f"  Scalars: {config.scalar_cols}")
    print(f"  Batch size: {args.batch_size}")

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    print(f"\n[2/5] Building {args.encoding} encoding model...")
    model = get_quantum_model(
        encoding=args.encoding,
        num_classes=num_classes,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_scalars=n_scalars,
        dropout=args.dropout,
    )
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_params = model.q_weights.numel()
    c_params = total_params - q_params
    print(f"  Total params:   {total_params:,}")
    print(f"  Quantum params: {q_params:,}  ({args.n_layers} layers × {args.n_qubits} qubits × 3)")
    print(f"  Classical params: {c_params:,}")

    # ------------------------------------------------------------------
    # 3. Loss, Optimizer, Scheduler
    # ------------------------------------------------------------------
    # Compute class weights for imbalanced classes
    train_labels = dm.train_ds.labels.numpy()
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=3)

    print(f"\n  Class weights: {class_weights.cpu().numpy().round(3)}")
    print(f"  Optimizer: Adam (lr={args.lr}, wd=1e-4)")
    print(f"  Scheduler: ReduceLROnPlateau (patience=3)")

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(f"\n[3/5] Training for {args.epochs} epochs...\n")

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc, _, _ = eval_epoch(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={lr_now:.2e}  time={elapsed:.1f}s")

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{encoding_name}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args),
                'class_names': class_names,
            }, ckpt_path)
            print(f"  ✓ New best val_acc={val_acc:.4f} — saved to {ckpt_path}")

    # ------------------------------------------------------------------
    # 5. Test evaluation
    # ------------------------------------------------------------------
    print(f"\n[4/5] Loading best checkpoint for test evaluation...")
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, f"best_{encoding_name}.pt"),
        map_location=DEVICE
    )
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, DEVICE
    )

    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.4f}")
    print(f"  Best Val Acc: {best_val_acc:.4f}")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=class_names, digits=4))

    # ------------------------------------------------------------------
    # 6. Save plots
    # ------------------------------------------------------------------
    print(f"\n[5/5] Saving results...")
    save_training_curves(history, encoding_name)
    save_confusion_matrix(test_labels, test_preds, class_names, encoding_name)

    # Save final summary
    summary_path = os.path.join(RESULTS_DIR, f"summary_{encoding_name}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Quantum SDSS Classifier — {encoding_name}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Encoding:    {args.encoding}\n")
        f.write(f"Qubits:      {args.n_qubits}\n")
        f.write(f"Layers:      {args.n_layers}\n")
        f.write(f"Epochs:      {args.epochs}\n")
        f.write(f"Batch size:  {args.batch_size}\n")
        f.write(f"LR:          {args.lr}\n\n")
        f.write(f"Total params:   {total_params:,}\n")
        f.write(f"Quantum params: {q_params:,}\n")
        f.write(f"Classical params: {c_params:,}\n\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
        f.write(f"Test Acc:     {test_acc:.4f}\n")
        f.write(f"Test Loss:    {test_loss:.4f}\n\n")
        f.write(f"Classification Report:\n")
        f.write(classification_report(test_labels, test_preds,
                                      target_names=class_names, digits=4))
    print(f"  Saved summary → {summary_path}")

    print(f"\n{'='*60}")
    print(f"  Done! Best val acc: {best_val_acc:.4f}  Test acc: {test_acc:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()