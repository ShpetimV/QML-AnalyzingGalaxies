"""
run_matrix_one.py  --  single-unit worker for the QCNN robustness matrix.

Trains exactly ONE (quadrant, model, seed) and writes:
    <run-dir>/<quadrant>/<model>/seed<S>/robustness.csv   (Sigma, Accuracy)
    <run-dir>/<quadrant>/<model>/seed<S>/jacobian.csv      (frob, spec per sample)
    <run-dir>/<quadrant>/<model>/seed<S>/plots/...         (history, confusion)
    <run-dir>/<quadrant>/<model>/seed<S>/worker.log

Parallelism is handled OUTSIDE this script by run_matrix.sh.
Aggregation (AUC, Wilcoxon, bands) is handled by aggregate.py.
"""
import sys, os, argparse, random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch

torch.set_num_threads(1)
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.param_config import SDSSDataConfig, TrainingConfig
from src.sdss_dataloader import SDSSDataModule
from src.training.trainer import SDSSPerformanceTrainer
from src.training.metrics import SDSSMetricTracker
from src.models.qcnn import (QCNNClassifier, HybridQCNNClassifier,
                             ClassicalTinyBaseline, ClassicalHugeBaseline)

NUM_CLASSES = 4
NOISE_LEVELS = [x / 10.0 for x in range(0, 41)]  # 0.0 .. 4.0 step 0.1

MODELS = {
    "QCNN": lambda: QCNNClassifier(NUM_CLASSES),
    "CNN_Tiny": lambda: ClassicalTinyBaseline(NUM_CLASSES),
    "CNN_Huge": lambda: ClassicalHugeBaseline(NUM_CLASSES),
    "QCNN_Hyb": lambda: HybridQCNNClassifier(NUM_CLASSES),  # encoding ablation (optional)
}
QUADRANTS = ["1_HighData_Clean", "2_HighData_Adv", "3_FewShot_Clean", "4_FewShot_Adv"]


# --------------------------------------------------------------------------- #
def set_seed(s):
    random.seed(s);
    np.random.seed(s);
    torch.manual_seed(s)


class _Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a", encoding="utf-8")

    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()

    def flush(self): self.terminal.flush(); self.log.flush()


class AdversarialNoiseWrapper(Dataset):
    def __init__(self, original_dataset, min_sigma=0.2, max_sigma=0.6):
        self.dataset = original_dataset
        self.min_sigma, self.max_sigma = min_sigma, max_sigma

    @property
    def labels(self): return self.dataset.labels

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx].copy()
        flux = item['flux']
        if torch.rand(1).item() > 0.5:
            return item
        sigma = torch.rand(1).item() * (self.max_sigma - self.min_sigma) + self.min_sigma
        noisy = flux + torch.randn_like(flux) * sigma * flux.std(dim=-1, keepdim=True)
        m = noisy.mean(dim=-1, keepdim=True)
        s = noisy.std(dim=-1, keepdim=True) + 1e-6
        item['flux'] = (noisy - m) / s
        return item


class InMemory(Dataset):
    """Iterate a dataset ONCE into RAM so every epoch is a tensor index, not a reload."""

    def __init__(self, ds):
        self.items = [ds[i] for i in range(len(ds))]
        self.labels = ds.labels
        self.is_train = getattr(ds, "is_train", True)

    def __len__(self): return len(self.items)

    def __getitem__(self, i): return self.items[i]


def _subset(dataset, idx):
    labels = dataset.labels.numpy()
    sub = Subset(dataset, idx)
    sub.labels = torch.as_tensor(labels[idx])
    sub.is_train = getattr(dataset, "is_train", True)
    return sub


def cap_per_class(dataset, cap):
    """Randomly keep <= cap samples per class (seeded by set_seed)."""
    labels = dataset.labels.numpy()
    idx = []
    for c in range(NUM_CLASSES):
        c_idx = np.where(labels == c)[0]
        if len(c_idx) > cap:
            c_idx = np.random.choice(c_idx, cap, replace=False)
        idx.extend(c_idx.tolist())
    np.random.shuffle(idx)
    return _subset(dataset, idx)


def slice_few_shot(dataset, n_per_class):
    """Randomly sample n per class (seeded) -- so seeds vary the subset, not just init."""
    labels = dataset.labels.numpy()
    idx = []
    for c in range(NUM_CLASSES):
        c_idx = np.where(labels == c)[0]
        idx.extend(np.random.choice(c_idx, min(n_per_class, len(c_idx)), replace=False).tolist())
    return _subset(dataset, idx)


def cap_eval(dataset, cap):
    """Deterministic first-`cap`-per-class subset, materialized. -> only used for big test sets."""
    labels = dataset.labels.numpy()
    idx = []
    for c in range(NUM_CLASSES):
        idx.extend(np.where(labels == c)[0][:cap].tolist())
    return InMemory(_subset(dataset, idx))


def build_quadrant(name, dm, cap):
    if name == "1_HighData_Clean": return InMemory(cap_per_class(dm.train_ds, cap))
    if name == "2_HighData_Adv":   return AdversarialNoiseWrapper(InMemory(cap_per_class(dm.train_ds, cap)))
    if name == "3_FewShot_Clean":  return InMemory(slice_few_shot(dm.train_ds, 20))
    if name == "4_FewShot_Adv":    return AdversarialNoiseWrapper(InMemory(slice_few_shot(dm.train_ds, 20)))
    raise ValueError(f"unknown quadrant {name}")


# --------------------------------------------------------------------------- #
def evaluate_with_noise(model, test_loader, device, sigma):
    model.eval();
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            flux = batch['flux'].to(device)
            labels = batch['label'].to(device)
            if sigma > 0.0:
                flux = flux + torch.randn_like(flux) * sigma * flux.std(dim=-1, keepdim=True)
                flux = (flux - flux.mean(dim=-1, keepdim=True)) / (flux.std(dim=-1, keepdim=True) + 1e-6)
            correct += (model(flux).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def jacobian_norms(model, test_loader, device, max_samples=128):
    """Per-sample ||d logits / d input||  (Frobenius and spectral). The mechanism
    probe: a smaller spectral norm == smaller local Lipschitz == more robust."""
    model.eval()
    xs = []
    for batch in test_loader:
        xs.append(batch['flux'])
        if sum(t.size(0) for t in xs) >= max_samples:
            break
    x = torch.cat(xs)[:max_samples].to(device).clone().requires_grad_(True)
    logits = model(x)
    B, C = logits.shape
    rows = [torch.autograd.grad(logits[:, c].sum(), x, retain_graph=True)[0].reshape(B, -1)
            for c in range(C)]
    J = torch.stack(rows, dim=1)  # [B, C, L]

    # Use CPU for SVD to avoid MPS limitations
    J_cpu = J.detach().cpu()
    frob = J_cpu.norm(dim=(1, 2)).numpy()
    spec = torch.linalg.svdvals(J_cpu).max(dim=1).values.numpy()
    return frob, spec


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quadrant", required=True, choices=QUADRANTS)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cap-per-class", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=120)  # 150 was overkill for tiny models
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--jac-samples", type=int, default=128)
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.run_dir) / a.quadrant / a.model / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Logger(out / "worker.log")
    print(f"[{a.quadrant} | {a.model} | seed {a.seed}] -> {out}")

    data_config = SDSSDataConfig(
        parquet_path=str(PROJECT_ROOT / "dataset" / "EXP4_QCNN_k1_g3_k4_g5_high_snr.parquet"),
        use_augmentation=False, batch_size=64, fixed_length=4096,
        scalar_cols=[], target_col="FINAL_CLASS",
    )
    dm = SDSSDataModule(data_config);
    dm.prepare_data()
    classes = list(dm.classes)
    val_loader = DataLoader(cap_eval(dm.val_ds, 75), batch_size=data_config.batch_size)
    test_loader = DataLoader(cap_eval(dm.test_ds, 5000), batch_size=data_config.batch_size)

    train_ds = build_quadrant(a.quadrant, dm, a.cap_per_class)

    FEWSHOT = a.quadrant[0] in ("3", "4")
    batch_size = len(train_ds) if FEWSHOT else 64  # full-batch GD on tiny data
    lr = 1e-3 if FEWSHOT else 2e-3
    n_restarts = 5 if (FEWSHOT and a.model.startswith("QCNN")) else 1

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    print(f"train samples: {len(train_ds)}")

    best_val, best_state, best_trainer = -1.0, None, None
    for r in range(n_restarts):
        set_seed(a.seed * 100 + r)  # distinct, reproducible init per restart
        model = MODELS[a.model]()
        train_cfg = TrainingConfig(epochs=a.epochs, lr=lr)
        trainer = SDSSPerformanceTrainer(model, train_cfg, run_name=f"{a.model}_r{r}",
                                         base_run_dir=str(out / f"restart{r}"))
        trainer.resume_file = str(out / f"restart{r}" / "resume.pt")
        trainer.train(train_loader, val_loader, train_cfg.epochs, lr)
        state = torch.load(os.path.join(trainer.checkpoint_dir, "best_model.pt"),
                           map_location=trainer.device)
        model.load_state_dict(state)
        v = evaluate_with_noise(model, val_loader, trainer.device, 0.0)  # clean val acc
        print(f"  restart {r}: val acc {v:.4f}")
        if v > best_val:
            best_val, best_state, best_trainer = v, state, trainer
    model.load_state_dict(best_state)
    trainer = best_trainer
    dev = trainer.device

    print(f"selected restart with val acc {best_val:.4f}")

    # diagnostics: history + confusion matrix on the clean test set
    tracker = SDSSMetricTracker(results_dir=trainer.plots_dir)
    tracker.plot_history(trainer.history)
    preds, labs = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            preds.extend(model(batch['flux'].to(dev)).argmax(1).cpu().numpy())
            labs.extend(batch['label'].numpy())
    tracker.plot_confusion_matrix(np.array(labs), np.array(preds),
                                  class_names=classes, split_name=a.model)

    # robustness decay
    print("[robustness decay]")
    accs = []
    for sigma in NOISE_LEVELS:
        acc = evaluate_with_noise(model, test_loader, dev, sigma)
        accs.append(acc);
        print(f"  sigma {sigma:.1f} -> {acc:.4f}")
    pd.DataFrame({"Sigma": NOISE_LEVELS, "Accuracy": accs}).to_csv(out / "robustness.csv", index=False)

    # Jacobian mechanism probe
    print("[jacobian probe]")
    try:
        frob, spec = jacobian_norms(model, test_loader, dev, a.jac_samples)
        pd.DataFrame({"frob": frob, "spec": spec}).to_csv(out / "jacobian.csv", index=False)
        print(f"  ||J||_2 mean={spec.mean():.4f}  ||J||_F mean={frob.mean():.4f}")
    except Exception as e:
        # Always emit a file so the aggregator can detect the failure explicitly
        pd.DataFrame({"frob": [np.nan], "spec": [np.nan]}).to_csv(out / "jacobian.csv", index=False)
        print(f"[warn] Jacobian probe failed: {e}")

    print("DONE")


if __name__ == "__main__":
    main()
