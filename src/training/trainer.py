import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from src.training.logger import get_global_logger


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # Ensure weight is a tensor if provided
        self.register_buffer('weight', weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Calculate unweighted cross entropy to get "clean" pt
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing
        )

        # 2. Get the probabilities (pt)
        pt = torch.exp(-ce_loss)

        # 3. Calculate the focal modulation
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Apply class weights manually
        if self.weight is not None:
            at = self.weight.gather(0, targets.data)
            focal_loss = focal_loss * at

        # 5. Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SDSSPerformanceTrainer:
    def __init__(self, model, config, run_name="Baseline_CNN"):

        self.config = config
        self.rng = np.random.default_rng(42)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        self.model = model.to(self.device)

        # Only use the scaler on CUDA (MPS handles it on its own)
        self.use_scaler = self.device.type == 'cuda'
        if self.use_scaler:
            self.scaler = torch.amp.GradScaler('cuda')

        # temporary resume file path
        self.resume_file = os.path.join("runs", f"resume_{run_name}.pt")

        # Check if we are recovering from a crashed run
        if os.path.exists(self.resume_file):
            print(f"Found active resume state for {run_name}. Restoring directories...")
            checkpoint_meta = torch.load(self.resume_file, map_location='cpu', weights_only=False)
            self.run_dir = checkpoint_meta['run_dir']
        else:
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

        # Class Weights for Focal Loss (class imbalance handling)
        train_labels = train_loader.dataset.full_labels[train_loader.dataset.indices]
        num_classes = self.model.classifier[-1].out_features
        class_counts = np.bincount(train_labels, minlength=num_classes)
        class_weights = 1.0 / (class_counts + 1e-5)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        self.logger.info("Calculated and applied inverse class weights to FocalLoss.")

        criterion = FocalLoss(
            weight=class_weights_tensor,
            gamma=2.0
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # OneCycleLR is great for rapid baselines
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
        )

        epochs_without_improvement = 0
        patience = 150

        start_epoch = 1

        # --- RESUME LOGIC ---
        if os.path.exists(self.resume_file):
            self.logger.info("Loading state from resume file to recover from interruption...")
            checkpoint = torch.load(self.resume_file, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if self.use_scaler and checkpoint['scaler_state_dict'] is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.history = checkpoint['history']
            epochs_without_improvement = checkpoint['epochs_without_improvement']

            self.logger.info(
                f"Successfully resumed! Starting at epoch {start_epoch} (Best Val Acc was: {self.best_val_acc:.4f})")

        self.logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(start_epoch, epochs + 1):
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

            # --- SAFE STATE SAVING ---
            resume_state = {
                'run_dir': self.run_dir,
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.use_scaler else None,
                'best_val_acc': self.best_val_acc,
                'history': self.history,
                'epochs_without_improvement': epochs_without_improvement
            }

            # Save to a temporary file first, then replace. Prevents corruption if server dies during save.
            tmp_file = self.resume_file + ".tmp"
            torch.save(resume_state, tmp_file)
            os.replace(tmp_file, self.resume_file)

            if epochs_without_improvement >= patience:
                self.logger.warning(
                    f"Early stopping triggered after {epoch} epochs. No improvement for {patience} epochs.")
                break

    def _run_epoch(self, loader, optimizer, scheduler, criterion, train=False):

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, total_samples = 0.0, 0
        acc_correct, acc_samples = 0, 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                # New DataLoader returns a dict
                flux = batch['flux'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                aux = batch['scalars'].to(self.device, non_blocking=True)

                if train:
                    optimizer.zero_grad()  # Clear gradients -> accurate tracking

                # --- MIXUP AUGMENTATION ---
                # only during train and with 50% chance to keep some original samples for stability
                amp_dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32

                with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                    if train and torch.rand(1).item() > 0.5:
                        alpha = 0.4  # Control strength of blending
                        lam = float(self.rng.beta(alpha, alpha))
                        rand_index = torch.randperm(flux.size(0)).to(self.device)

                        # Blend the input flux and the auxiliary scalars
                        mixed_flux = lam * flux + (1 - lam) * flux[rand_index]
                        mixed_aux = lam * aux + (1 - lam) * aux[rand_index]

                        logits = self.model(mixed_flux, mixed_aux)
                        loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[rand_index])

                        # dont track accuracy for mixed since it leads to unusable train acc
                        total_loss += loss.item() * labels.size(0)
                        total_samples += labels.size(0)
                    else:
                        # original forward pass
                        logits = self.model(flux, aux)
                        loss = criterion(logits, labels)

                        # track loss AND accuracy
                        total_loss += loss.item() * labels.size(0)
                        total_samples += labels.size(0)

                        acc_correct += (logits.argmax(1) == labels).sum().item()
                        acc_samples += labels.size(0)

                # --- CROSS-PLATFORM BACKWARD PASS ---
                if train:
                    if self.use_scaler:
                        # Server (CUDA) Path: Unscale gradients before clipping
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        # Mac (MPS) Path: Standard backward pass
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()

                    scheduler.step()

        epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc = acc_correct / acc_samples if acc_samples > 0 else 0.0

        return epoch_loss, epoch_acc
