import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from src.training.logger import get_global_logger


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight  # Class weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard cross entropy
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none',
                                              label_smoothing=self.label_smoothing)

        # Get the probabilities of the true classes
        pt = torch.exp(-ce_loss)

        # Apply the focal loss formula
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SDSSPerformanceTrainer:
    def __init__(self, model, config, run_name="Baseline_CNN"):

        self.config = config

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("runs", f"{run_name}_{timestamp}")
        self.checkpoint_dir = os.path.join(self.run_dir, "trained_models")
        self.plots_dir = os.path.join(self.run_dir, "plots")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # 3. Initialize Global Logger
        log_path = os.path.join(self.run_dir, "training.log")
        self.logger = get_global_logger(run_name, log_file=log_path)
        self.logger.info(f"Initialized {run_name} on {self.device}")

        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train(self, train_loader, val_loader, epochs, lr=3e-4, weight_decay=1e-4):
        criterion = FocalLoss(
            weight=None,
            gamma=2.0
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # OneCycleLR is great for rapid baselines
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
        )

        epochs_without_improvement = 0
        patience = 100

        self.logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, optimizer, scheduler, criterion, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, None, None, criterion, train=False)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            self.logger.info(
                f"Epoch {epoch:02d}/{epochs} | Tr Loss: {train_loss:.4f} | Tr Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                epochs_without_improvement = 0

                save_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(f" -> New Best Model Saved (Acc: {val_acc:.4f})")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                self.logger.warning(
                    f"Early stopping triggered after {epoch} epochs. No improvement for {patience} epochs.")
                break

    def _run_epoch(self, loader, optimizer, scheduler, criterion, train=False):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                # New DataLoader returns a dict
                flux = batch['flux'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                aux = batch['scalars'].to(self.device, non_blocking=True)

                logits = self.model(flux, aux)
                loss = criterion(logits, labels)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total
