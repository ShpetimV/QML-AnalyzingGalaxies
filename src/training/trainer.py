import os
import torch
import torch.nn as nn
from tqdm import tqdm


class SDSSPerformanceTrainer:
    def __init__(self, model, config, device=None):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.config = config
        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train(self, train_loader, val_loader, epochs, lr=3e-4, weight_decay=1e-4, class_weights=None):
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # OneCycleLR is great for rapid baselines
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
        )

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, optimizer, scheduler, criterion, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, None, None, criterion, train=False)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(
                f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f}/{val_loss:.4f} | Acc: {train_acc:.4f}/{val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_baseline_model.pt")

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