import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from astropy.io import fits
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

from model_subclasses import SpectraClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_DIR    = "../dataset/gasnet"
CHECKPOINT_DIR = "./checkpoints_gasnet"
RESULTS_DIR    = "./results_gasnet"

HF_REPO        = "Fucheng/GaSNet-II-SDSS-dataset"
FIXED_LENGTH   = 3600    # compact spectra used in GaSNet-II paper
NUM_WORKERS    = 4
BATCH_SIZE     = 256
EPOCHS         = 30
LR             = 3e-4
WEIGHT_DECAY   = 1e-4

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

os.makedirs(DATASET_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)


# ---------------------------------------------------------------------------
# Download FITS files from HuggingFace
# ---------------------------------------------------------------------------

def download_fits():
    files = ["train.fits", "valid.fits", "test.fits"]
    paths = {}
    for fname in files:
        local = os.path.join(DATASET_DIR, fname)
        if os.path.exists(local):
            print(f"  {fname} already exists, skipping download")
        else:
            print(f"  Downloading {fname} ...")
            local = hf_hub_download(
                repo_id=HF_REPO,
                filename=fname,
                repo_type="dataset",
                local_dir=DATASET_DIR,
            )
        paths[fname] = local
    return paths


# ---------------------------------------------------------------------------
# FITS inspection + loading
# ---------------------------------------------------------------------------

def inspect_fits(path):
    """Print the HDU structure so we know what columns to use."""
    with fits.open(path) as hdul:
        hdul.info()
        for i, hdu in enumerate(hdul):
            if hasattr(hdu, 'columns') and hdu.columns:
                print(f"\nHDU {i} columns: {hdu.columns.names}")
            elif hdu.data is not None:
                print(f"\nHDU {i} data shape: {hdu.data.shape}, dtype: {hdu.data.dtype}")


def load_fits(path, label_map=None):
    """
    Load flux arrays and labels from a GaSNet-II FITS file.
    Returns (flux_array, label_array, label_map).

    The file has a BinTable with columns for flux and class/subclass.
    We inspect on first load to discover column names.
    """
    with fits.open(path, memmap=False) as hdul:
        # Find the table HDU
        table_hdu = None
        for hdu in hdul:
            if hasattr(hdu, 'columns') and hdu.columns:
                table_hdu = hdu
                break

        if table_hdu is None:
            # Fallback: primary HDU is an image array + separate label HDU
            data   = hdul[0].data     # shape (N, 3600)
            labels_raw = hdul[1].data if len(hdul) > 1 else None
        else:
            cols = [c.name.upper() for c in table_hdu.columns]
            print(f"  Available columns: {cols}")

            # Flux column — try common names
            flux_col = next((c for c in cols if 'FLUX' in c or 'SPEC' in c or 'DATA' in c), None)
            if flux_col is None:
                # Last resort: first numeric array column
                for c in table_hdu.columns:
                    if 'E' in c.format or 'D' in c.format or 'J' in c.format:
                        if table_hdu.data[c.name].ndim > 1:
                            flux_col = c.name.upper()
                            break

            # Class / label column
            class_col    = next((c for c in cols if c in ('CLASS', 'LABEL', 'TYPE')),   None)
            subclass_col = next((c for c in cols if c in ('SUBCLASS', 'SUBTYPE', 'SUB')), None)

            print(f"  Using flux column:     {flux_col}")
            print(f"  Using class column:    {class_col}")
            print(f"  Using subclass column: {subclass_col}")

            data = np.array(table_hdu.data[flux_col], dtype=np.float32)

            # Build class string: "STAR_K5", "GALAXY_STARFORMING", "QSO" etc.
            if class_col and subclass_col:
                raw_classes = []
                for row in table_hdu.data:
                    cls = str(row[class_col]).strip().strip("[]'\" ")
                    sub = str(row[subclass_col]).strip().strip("[]'\" ")
                    if sub and sub.lower() not in ('', 'none', 'nan'):
                        raw_classes.append(f"{cls}_{sub}")
                    else:
                        raw_classes.append(cls)
            elif class_col:
                raw_classes = [str(r[class_col]).strip() for r in table_hdu.data]
            else:
                raise ValueError("Could not find a class/label column in the FITS file.")

            labels_raw = np.array(raw_classes)

    # Build label map from the training set (reuse if already built)
    if label_map is None:
        unique = sorted(set(labels_raw))
        label_map = {cls: i for i, cls in enumerate(unique)}
        print(f"\n  Found {len(label_map)} classes:")
        for cls, idx in label_map.items():
            count = (labels_raw == cls).sum()
            print(f"    {idx:2d}  {cls:30s}  {count:,} samples")

    label_ints = np.array([label_map[c] for c in labels_raw], dtype=np.int64)

    return data, label_ints, label_map


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GaSNetDataset(Dataset):
    def __init__(self, flux_data, labels, fixed_length=FIXED_LENGTH):
        self.flux_data    = flux_data
        self.labels       = labels
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        flux = self.flux_data[idx].astype(np.float32)

        # Pad or truncate
        if len(flux) >= self.fixed_length:
            flux = flux[:self.fixed_length]
        else:
            flux = np.pad(flux, (0, self.fixed_length - len(flux)), mode='constant')

        # Replace NaN/Inf with 0
        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)

        # Robust per-sample normalisation
        median = np.median(flux)
        iqr    = np.percentile(flux, 75) - np.percentile(flux, 25)
        flux   = (flux - median) / iqr if iqr > 0 else flux - median
        flux   = np.clip(flux, -10, 10)

        return (
            torch.tensor(flux, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for flux, labels in loader:
        flux, labels = flux.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(flux, aux=None)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for flux, labels in loader:
        flux, labels = flux.to(device), labels.to(device)
        logits = model(flux, aux=None)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)
        total_loss += loss.item() * labels.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        if return_preds:
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    loss_avg = total_loss / total
    acc      = correct / total
    if return_preds:
        return loss_avg, acc, np.array(all_preds), np.array(all_labels)
    return loss_avg, acc


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_training_plot(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0d0d1a')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], color='#00BFFF', label='Train')
    ax1.plot(epochs, history['val_loss'],   color='#FFD700', label='Val')
    ax1.set_title('Loss', color='white')
    ax1.set_xlabel('Epoch', color='#888888')
    ax1.legend()
    ax2.plot(epochs, history['train_acc'], color='#00BFFF', label='Train')
    ax2.plot(epochs, history['val_acc'],   color='#FFD700', label='Val')
    ax2.set_title('Accuracy', color='white')
    ax2.set_xlabel('Epoch', color='#888888')
    ax2.legend()
    fig.suptitle('Training History — GaSNet-II dataset', color='white', fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves_gasnet.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved training curves → {path}")
    plt.close()


def save_confusion_matrix(preds, labels, idx_to_label, split_name='test'):
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    n  = len(class_names)
    cm = confusion_matrix(labels, preds, labels=list(range(n)))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm.astype(float), row_sums, where=row_sums != 0)

    fig_size = max(12, n * 0.75)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.tick_params(colors='#888888')
    cbar.set_label('Recall (row-normalised)', color='#888888', fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9, color='#cccccc')
    ax.set_yticklabels(class_names, fontsize=9, color='#cccccc')

    thresh = 0.5
    for i, j in itertools.product(range(n), range(n)):
        val = cm_norm[i, j]
        if val > 0.005:
            color = 'white' if val > thresh else '#aaaaaa'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    ax.set_ylabel('True label',      color='#cccccc', fontsize=11)
    ax.set_xlabel('Predicted label', color='#cccccc', fontsize=11)
    ax.set_title(f'Confusion matrix — {split_name} set (GaSNet-II data)',
                 color='white', fontsize=12, pad=14)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_matrix_{split_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved confusion matrix  → {path}")
    plt.close()


def save_per_class_accuracy(preds, labels, idx_to_label, split_name='test'):
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    n = len(class_names)
    per_class = [(class_names[i], (preds[labels == i] == i).mean() if (labels == i).sum() > 0 else 0.0)
                 for i in range(n)]
    per_class.sort(key=lambda x: x[1], reverse=True)
    names, accs = zip(*per_class)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.7), 5))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    colors = ['#00BFFF' if a >= 0.8 else '#FFD700' if a >= 0.5 else '#FF6347' for a in accs]
    ax.bar(names, accs, color=colors, edgecolor='#333333', linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color='#00BFFF', linewidth=0.6, linestyle='--', alpha=0.4)
    ax.set_ylabel('Accuracy', color='#888888')
    ax.set_title(f'Per-class accuracy — {split_name} set (GaSNet-II data)', color='white', fontsize=12)
    ax.tick_params(colors='#888888')
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9, color='#cccccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"per_class_accuracy_{split_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved per-class accuracy → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Using device: {DEVICE}")
    print("Dataset: GaSNet-II (Fucheng/GaSNet-II-SDSS-dataset)\n")

    # 1. Download
    print("Downloading FITS files from HuggingFace...")
    paths = download_fits()

    # 2. Inspect structure of train.fits once
    print(f"\nInspecting {paths['train.fits']}...")
    inspect_fits(paths['train.fits'])

    # 3. Load all splits
    print(f"\nLoading train.fits...")
    train_flux, train_labels, label_map = load_fits(paths['train.fits'])

    idx_to_label = {v: k for k, v in label_map.items()}
    num_classes  = len(label_map)

    print(f"\nLoading valid.fits...")
    val_flux, val_labels, _ = load_fits(paths['valid.fits'], label_map=label_map)

    print(f"\nLoading test.fits...")
    test_flux, test_labels, _ = load_fits(paths['test.fits'], label_map=label_map)

    print(f"\nLoaded:  train={len(train_labels):,}  val={len(val_labels):,}  test={len(test_labels):,}")
    print(f"Classes: {num_classes}")

    # 4. DataLoaders
    pin = (DEVICE == "cuda")

    train_ds = GaSNetDataset(train_flux, train_labels)
    val_ds   = GaSNetDataset(val_flux,   val_labels)
    test_ds  = GaSNetDataset(test_flux,  test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # 5. Model — num_classes set from actual data
    model = SpectraClassifier(
        num_classes=num_classes,
        input_length=FIXED_LENGTH,
        aux_features=0,
        dropout=0.3,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # Dataset is balanced (20k per class) → no class weighting needed
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            scheduler, criterion, DEVICE)
        val_loss, val_acc     = eval_epoch(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model_gasnet.pt")
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':          val_acc,
                'label_map':        label_map,
                'idx_to_label':     idx_to_label,
                'num_classes':      num_classes,
            }, ckpt_path)
            print(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved")

    # 6. Test evaluation
    print(f"\nLoading best checkpoint...")
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model_gasnet.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, preds, labels_out = eval_epoch(
        model, test_loader, criterion, DEVICE, return_preds=True
    )
    print(f"\nTest results:  loss={test_loss:.4f}  acc={test_acc:.4f}")
    print(f"Best val acc:  {best_val_acc:.4f}")

    # 7. Save plots
    save_training_plot(history)
    save_confusion_matrix(preds, labels_out, idx_to_label, split_name='test')
    save_per_class_accuracy(preds, labels_out, idx_to_label, split_name='test')

    _, _, val_preds, val_labels_out = eval_epoch(
        model, val_loader, criterion, DEVICE, return_preds=True
    )
    save_confusion_matrix(val_preds, val_labels_out, idx_to_label, split_name='val')

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()