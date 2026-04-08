import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt

from model import SpectraClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_PATH  = "../dataset/sdss_merged_full.parquet"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR    = "./results"

FIXED_LENGTH  = 3522       # pad/truncate all spectra to this length
AUX_FEATURES  = 7          # Z, VDISP, SPECTROFLUX_U/G/R/I/Z
NUM_CLASSES   = 3          # STAR, GALAXY, QSO

BATCH_SIZE    = 256
EPOCHS        = 30
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4

# Split: 70% train / 15% val / 15% test
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST_RATIO  = 0.15  (remainder)

DEVICE = (
    "mps"   if torch.backends.mps.is_available() else
    "cuda"  if torch.cuda.is_available()          else
    "cpu"
)
print(f"Using device: {DEVICE}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CLASS_MAP = {'STAR': 0, 'GALAXY': 1, 'QSO': 2}

class SDSSDataset(Dataset):
    def __init__(self, df, fixed_length=FIXED_LENGTH):
        self.df           = df.reset_index(drop=True)
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Flux: pad or truncate to fixed length ---
        flux = np.array(row['flux'], dtype=np.float32)
        if len(flux) >= self.fixed_length:
            flux = flux[:self.fixed_length]
        else:
            flux = np.pad(flux, (0, self.fixed_length - len(flux)), mode='constant')

        # --- Per-sample normalization (robust: median + IQR) ---
        median = np.median(flux)
        iqr    = np.percentile(flux, 75) - np.percentile(flux, 25)
        if iqr > 0:
            flux = (flux - median) / iqr
        else:
            flux = flux - median

        # Clip extreme outliers (cosmic rays etc.)
        flux = np.clip(flux, -10, 10)

        # --- Auxiliary features ---
        aux = np.array([
            row['Z']              if pd.notna(row['Z'])              else 0.0,
            row['VDISP']          if pd.notna(row['VDISP'])          else 0.0,
            row['SPECTROFLUX_U']  if pd.notna(row['SPECTROFLUX_U'])  else 0.0,
            row['SPECTROFLUX_G']  if pd.notna(row['SPECTROFLUX_G'])  else 0.0,
            row['SPECTROFLUX_R']  if pd.notna(row['SPECTROFLUX_R'])  else 0.0,
            row['SPECTROFLUX_I']  if pd.notna(row['SPECTROFLUX_I'])  else 0.0,
            row['SPECTROFLUX_Z']  if pd.notna(row['SPECTROFLUX_Z'])  else 0.0,
        ], dtype=np.float32)

        # Normalize aux features simply
        aux = np.clip(aux, -1e6, 1e6)
        aux = aux / (np.abs(aux).max() + 1e-8)

        label = CLASS_MAP[row['class']]

        return (
            torch.tensor(flux, dtype=torch.float32),
            torch.tensor(aux,  dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_and_split():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Total rows: {len(df):,}")

    # Keep only clean, labeled rows
    df = df[df['class'].isin(['STAR', 'GALAXY', 'QSO'])]
    df = df[df['ZWARNING'] == False]
    df = df[df['snMedian'] >= 5]
    print(f"After filtering: {len(df):,} rows")
    print("Class distribution:")
    print(df['class'].value_counts().to_string())

    # --- Stratified split to keep class balance in each set ---
    # First split off 70% train
    train_df, temp_df = train_test_split(
        df, test_size=(1 - TRAIN_RATIO),
        stratify=df['class'], random_state=42
    )

    # Split remaining 30% into val (15%) and test (15%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df['class'], random_state=42
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,}   ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,}  ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Class weights (handle imbalance)
# ---------------------------------------------------------------------------

def get_class_weights(train_df):
    counts = train_df['class'].map(CLASS_MAP).value_counts().sort_index()
    total  = counts.sum()
    weights = total / (NUM_CLASSES * counts)
    print(f"\nClass weights: {weights.values}")
    return torch.tensor(weights.values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training & validation steps
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for flux, aux, labels in loader:
        flux, aux, labels = flux.to(device), aux.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(flux, aux)
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
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for flux, aux, labels in loader:
        flux, aux, labels = flux.to(device), aux.to(device), labels.to(device)

        logits = model(flux, aux)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Plot training curves
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

    fig.suptitle('Training History', color='white', fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved training curves to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load data
    train_df, val_df, test_df = load_and_split()

    train_ds = SDSSDataset(train_df)
    val_ds   = SDSSDataset(val_df)
    test_ds  = SDSSDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # 2. Model
    model = SpectraClassifier(
        num_classes=NUM_CLASSES,
        input_length=FIXED_LENGTH,
        aux_features=AUX_FEATURES,
        dropout=0.3,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # 3. Loss with class weights
    class_weights = get_class_weights(train_df).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Optimizer + OneCycleLR scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,           # 10% warmup
        anneal_strategy='cos',
    )

    # 5. Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            scheduler, criterion, DEVICE)
        val_loss,   val_acc   = eval_epoch(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':    val_acc,
                'val_loss':   val_loss,
            }, ckpt_path)
            print(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved")

    # 6. Final test evaluation
    print(f"\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"\nTest results:  loss={test_loss:.4f}  acc={test_acc:.4f}")
    print(f"Best val acc:  {best_val_acc:.4f}")

    # 7. Save training curves
    save_training_plot(history)


if __name__ == "__main__":
    main()