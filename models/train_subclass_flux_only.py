import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools

from model import SpectraClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_PATH   = "../dataset/sdss_merged_full.parquet"
CHECKPOINT_DIR = "./checkpoints_subclass"
RESULTS_DIR    = "./results_subclass"

FIXED_LENGTH   = 3522
MIN_SAMPLES    = 150    # drop any subclass with fewer samples than this
BATCH_SIZE     = 256
EPOCHS         = 30
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 4
TRAIN_RATIO    = 0.70

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)


# ---------------------------------------------------------------------------
# Subclass normalisation + grouping
# ---------------------------------------------------------------------------

def normalise_subclass(row):
    """
    Maps raw subClass values to clean grouped labels.
    Returns None for rows that should be dropped.
    """
    main  = str(row['class']).strip().upper()
    sub   = str(row['subClass']).strip() if pd.notna(row['subClass']) else ''

    # ------------------------------------------------------------------ STARS
    if main == 'STAR':
        # Normalise luminosity/peculiarity suffixes: M0V → M0, A0p → A0
        import re
        sub_clean = re.sub(r'[Vvp+\-].*$', '', sub).strip()

        # Merge very-late M dwarfs
        if sub_clean in ('M7', 'M8', 'M9'):
            return 'M_late'

        # Keep individual well-populated types
        if sub_clean in (
            'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
            'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7',
            'F2', 'F5', 'F6', 'F7', 'F8', 'F9',
            'G0', 'G1', 'G2',
            'A0',
        ):
            return sub_clean

        # Merge all white dwarf variants
        if sub_clean in ('WD', 'WDmagnetic', 'CarbonWD'):
            return 'WD'

        # Drop rare / untrainable star types
        return None

    # --------------------------------------------------------------- GALAXIES
    if main == 'GALAXY':
        if not sub:
            return None   # 41k unlabelled galaxies — exclude

        # Merge BROADLINE variants back to their parent
        if 'STARFORMING' in sub:
            return 'STARFORMING'
        if 'STARBURST' in sub:
            return 'STARBURST'
        if 'AGN' in sub:
            return 'AGN'
        if sub == 'BROADLINE':
            return 'GALAXY_BROADLINE'

        return None   # anything else is too rare

    # ------------------------------------------------------------------- QSOs
    if main == 'QSO':
        return 'QSO'   # all QSO subtypes merged

    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SDSSSubclassDataset(Dataset):
    def __init__(self, df, label_map, fixed_length=FIXED_LENGTH):
        self.df          = df.reset_index(drop=True)
        self.label_map   = label_map
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        flux = np.array(row['flux'], dtype=np.float32)
        if len(flux) >= self.fixed_length:
            flux = flux[:self.fixed_length]
        else:
            flux = np.pad(flux, (0, self.fixed_length - len(flux)), mode='constant')

        # Robust per-sample normalisation
        median = np.median(flux)
        iqr    = np.percentile(flux, 75) - np.percentile(flux, 25)
        flux   = (flux - median) / iqr if iqr > 0 else flux - median
        flux   = np.clip(flux, -10, 10)

        label = self.label_map[row['subclass_grouped']]

        return (
            torch.tensor(flux,  dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Data loading, grouping, splitting
# ---------------------------------------------------------------------------

def load_and_prepare():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Total rows: {len(df):,}")

    # Quality filters
    df = df[df['ZWARNING'] == False]
    df = df[df['snMedian'] >= 5]

    # Apply subclass grouping
    print("Applying subclass grouping...")
    df['subclass_grouped'] = df.apply(normalise_subclass, axis=1)

    # Drop rows with no valid subclass
    df = df[df['subclass_grouped'].notna()].copy()
    print(f"After grouping: {len(df):,} rows")

    # Show raw counts before filtering
    counts = df['subclass_grouped'].value_counts()
    print(f"\nSubclass counts (before MIN_SAMPLES filter):")
    print(counts.to_string())

    # Drop classes with too few samples
    valid_classes = counts[counts >= MIN_SAMPLES].index.tolist()
    df = df[df['subclass_grouped'].isin(valid_classes)].copy()

    counts_final = df['subclass_grouped'].value_counts()
    print(f"\nFinal subclass counts ({len(valid_classes)} classes, MIN_SAMPLES={MIN_SAMPLES}):")
    print(counts_final.to_string())
    print(f"\nTotal usable rows: {len(df):,}")

    # Build label map (alphabetical for reproducibility)
    sorted_classes = sorted(valid_classes)
    label_map      = {cls: i for i, cls in enumerate(sorted_classes)}
    idx_to_label   = {i: cls for cls, i in label_map.items()}

    print(f"\nLabel map:")
    for cls, idx in sorted(label_map.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}  {cls}  ({counts_final[cls]:,} samples)")

    # Stratified split
    train_df, temp_df = train_test_split(
        df, test_size=(1 - TRAIN_RATIO),
        stratify=df['subclass_grouped'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df['subclass_grouped'], random_state=42
    )

    print(f"\nSplit: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    return train_df, val_df, test_df, label_map, idx_to_label


# ---------------------------------------------------------------------------
# Class weights (handles heavy imbalance between e.g. STARFORMING vs M_late)
# ---------------------------------------------------------------------------

def get_class_weights(train_df, label_map):
    num_classes = len(label_map)
    counts = train_df['subclass_grouped'].map(label_map).value_counts().sort_index()
    # Fill any missing class indices with 1 to avoid division by zero
    counts = counts.reindex(range(num_classes), fill_value=1)
    weights = counts.sum() / (num_classes * counts)
    print(f"\nClass weights (min={weights.min():.2f}, max={weights.max():.2f})")
    return torch.tensor(weights.values, dtype=torch.float32)


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

        preds = logits.argmax(1)
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

    fig.suptitle('Training History — Subclass', color='white', fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves_subclass.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved training curves → {path}")
    plt.close()


def save_confusion_matrix(preds, labels, idx_to_label, split_name='test'):
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    n = len(class_names)

    cm = confusion_matrix(labels, preds, labels=list(range(n)))

    # Normalise rows → recall per class
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_norm, row_sums, where=row_sums != 0)

    # Figure size scales with number of classes
    fig_size = max(14, n * 0.7)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    im = ax.imshow(cm_norm, interpolation='nearest',
                   cmap=plt.cm.Blues, vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.tick_params(colors='#888888')
    cbar.set_label('Recall (row-normalised)', color='#888888', fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right',
                       fontsize=9, color='#cccccc')
    ax.set_yticklabels(class_names, fontsize=9, color='#cccccc')

    # Annotate cells
    thresh = 0.5
    for i, j in itertools.product(range(n), range(n)):
        val = cm_norm[i, j]
        if val > 0.005:     # skip near-zero cells for cleanliness
            color = 'white' if val > thresh else '#aaaaaa'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    ax.set_ylabel('True label', color='#cccccc', fontsize=11)
    ax.set_xlabel('Predicted label', color='#cccccc', fontsize=11)
    ax.set_title(f'Confusion matrix — {split_name} set (row-normalised recall)',
                 color='white', fontsize=12, pad=14)

    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_matrix_{split_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved confusion matrix  → {path}")
    plt.close()


def save_per_class_accuracy(preds, labels, idx_to_label, split_name='test'):
    """Bar chart of per-class accuracy, sorted descending."""
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    n = len(class_names)

    per_class_acc = []
    for i in range(n):
        mask = labels == i
        if mask.sum() > 0:
            per_class_acc.append((class_names[i], (preds[mask] == i).mean()))
        else:
            per_class_acc.append((class_names[i], 0.0))

    per_class_acc.sort(key=lambda x: x[1], reverse=True)
    names, accs = zip(*per_class_acc)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.55), 5))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    colors = ['#00BFFF' if a >= 0.8 else '#FFD700' if a >= 0.5 else '#FF6347'
              for a in accs]
    ax.bar(names, accs, color=colors, edgecolor='#333333', linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy', color='#888888')
    ax.set_xlabel('Subclass', color='#888888')
    ax.set_title(f'Per-class accuracy — {split_name} set', color='white', fontsize=12)
    ax.tick_params(colors='#888888')
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9, color='#cccccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.axhline(0.8, color='#00BFFF', linewidth=0.6, linestyle='--', alpha=0.4)

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
    print("Mode: SUBCLASS training\n")

    train_df, val_df, test_df, label_map, idx_to_label = load_and_prepare()

    num_classes = len(label_map)
    print(f"\nTraining on {num_classes} subclasses")

    pin = (DEVICE == "cuda")

    train_ds = SDSSSubclassDataset(train_df, label_map)
    val_ds   = SDSSSubclassDataset(val_df,   label_map)
    test_ds  = SDSSSubclassDataset(test_df,  label_map)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    model = SpectraClassifier(
        num_classes=num_classes,
        input_length=FIXED_LENGTH,
        aux_features=0,
        dropout=0.3,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    class_weights = get_class_weights(train_df, label_map).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

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
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model_subclass.pt")
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':          val_acc,
                'val_loss':         val_loss,
                'label_map':        label_map,
                'idx_to_label':     idx_to_label,
                'num_classes':      num_classes,
            }, ckpt_path)
            print(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved")

    # --- Final evaluation on test set ---
    print(f"\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "best_model_subclass.pt"),
        map_location=DEVICE
    )
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, preds, labels = eval_epoch(
        model, test_loader, criterion, DEVICE, return_preds=True
    )
    print(f"\nTest results:  loss={test_loss:.4f}  acc={test_acc:.4f}")
    print(f"Best val acc:  {best_val_acc:.4f}")

    # --- Save all plots ---
    save_training_plot(history)
    save_confusion_matrix(preds, labels, idx_to_label, split_name='test')
    save_per_class_accuracy(preds, labels, idx_to_label, split_name='test')

    # Also save val confusion matrix
    _, _, val_preds, val_labels = eval_epoch(
        model, val_loader, criterion, DEVICE, return_preds=True
    )
    save_confusion_matrix(val_preds, val_labels, idx_to_label, split_name='val')

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()