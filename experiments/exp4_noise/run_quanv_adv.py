import sys
import os
import random
from pathlib import Path
import datetime

# Force single-threaded PyTorch for parallelism
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)

# make seeds/head CLI-selectable
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--head", choices=["heavy", "light"], required=True)
parser.add_argument("--seeds", type=int, nargs="+", required=True)
parser.add_argument("--run-dir", type=str, required=True)   # created ONCE by launcher
args = parser.parse_args()

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import math
from torch.autograd.functional import jacobian
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
EXP_NAME = "adv"
SPLIT_SEED = 42

from src.param_config import SDSSDataConfig, TrainingConfig
from src.sdss_dataloader import SDSSDataModule
from src.training.trainer import SDSSPerformanceTrainer
from src.models.quanvolution import QuanvClassifier, ClassicalConvClassifier
import src.models.quanvolution as qv

# --- CONFIGURATION ---
NUM_CLASSES = 2
SEQ_LEN = 1024
# Sigma values from 0.0 to 4.0 in increments of 0.1
NOISE_LEVELS = [x / 10.0 for x in range(0, 41)]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# Create a master timestamped folder for THIS specific run
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_RUN_DIR = PROJECT_ROOT / "experiment_results" / "experiment4_quanv_noise_robustness" / "adversarial_training" / f"Run_{TIMESTAMP}"

class InMemory(Dataset):
    def __init__(self, ds):
        self.items = [ds[i] for i in range(len(ds))]   # clean, cached once
        self.labels = ds.labels
        self.is_train = getattr(ds, "is_train", True)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

# ---------------------------------------------------------------------------
# JACOBIAN-NORM PROBE
# ---------------------------------------------------------------------------
def make_quanv_fn(quanv):
    """Per-patch map for the quantum filter: R^4 -> R^12 (incl. shared encoding)."""
    def fn(x4):                                   # x4: (4,)
        p = x4.unsqueeze(0)                        # (1,4)
        n = torch.norm(p, p=2, dim=1, keepdim=True) + 1e-8
        p = (p / n) * (math.pi / 2.0)              # spherical encoding (shared)
        p = p * torch.sigmoid(quanv.input_squeeze) # quantum attention
        out = quanv.qnode(p.cpu(), quanv.q_weights.cpu(), quanv.lens_weights.cpu())
        return torch.stack(out, dim=1).float().squeeze(0)   # (12,)
    return fn

def make_classical_fn(model):
    """Per-patch map for the classical filter: R^4 -> R^12 (same encoding)."""
    def fn(x4):
        p = x4.unsqueeze(0)
        n = torch.norm(p, p=2, dim=1, keepdim=True) + 1e-8
        p = (p / n) * (math.pi / 2.0)
        return model.filter(p).squeeze(0)          # (12,)
    return fn

def sample_test_patches(test_loader, seq_len, kernel_size, stride, n=300, norm_floor_pct=10):
    batch = next(iter(test_loader))
    flux = F.adaptive_avg_pool1d(batch['flux'], seq_len)
    patches = flux.unfold(2, kernel_size, stride).reshape(-1, kernel_size)
    norms = patches.norm(dim=1)
    patches = patches[norms > torch.quantile(norms, norm_floor_pct / 100.0)]  # kill the 1/||p|| singularity
    idx = torch.randperm(patches.size(0))[:n]
    return patches[idx]

def mean_jac_norm(fn, patches, use_operator_norm=False):
    vals = []
    for i in range(patches.size(0)):
        J = jacobian(fn, patches[i].detach().clone())
        v = torch.linalg.svdvals(J).max().item() if use_operator_norm else J.norm().item()
        vals.append(v)
    vals = np.array(vals)
    return float(np.median(vals)), float(np.subtract(*np.percentile(vals, [75, 25])))  # median, IQR                           # same patches used for BOTH models

def run_jacobian_probe(quanv_model, cnn_model, test_loader, n=300):
    quanv_model.eval().cpu(); cnn_model.eval().cpu()
    patches = sample_test_patches(test_loader, SEQ_LEN, 4, 2, n=n)
    qm, qs = mean_jac_norm(make_quanv_fn(quanv_model.quanv), patches)
    cm, cs = mean_jac_norm(make_classical_fn(cnn_model), patches)
    print(f"[Jacobian] Quanv ||J||_F = {qm:.4f} ± {qs:.4f}")
    print(f"[Jacobian] CNN   ||J||_F = {cm:.4f} ± {cs:.4f}")
    print(f"[Jacobian] ratio Quanv/CNN = {qm/cm:.3f}  (<1 => quantum map is smoother)")
    return qm, cm

# ---------------------------------------------------------------------------
# ADVERSARIAL DATASET WRAPPER
# ---------------------------------------------------------------------------
class AdversarialNoiseWrapper(Dataset):
    """
    Dynamically injects Continuous Uniform Gaussian Noise into the training data.
    This forces the model to learn robust features rather than memorizing a specific noise frequency.
    """

    def __init__(self, original_dataset, min_sigma=0.1, max_sigma=0.6):
        self.dataset = original_dataset
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    @property
    def labels(self):
        """Expose the underlying dataset labels for the Trainer's Focal Loss calculation."""
        return self.dataset.labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Fetch the clean item
        item = self.dataset[idx].copy()
        flux = item['flux']

        # show clean sample in 50% of the cases
        if torch.rand(1).item() > 0.5:
            return item

        # otherwise: Sample a random sigma for this specific sample
        sigma = (torch.rand(1).item() * (self.max_sigma - self.min_sigma)) + self.min_sigma

        # Inject Noise
        noise = torch.randn_like(flux) * sigma * flux.std(dim=-1, keepdim=True)
        noisy_flux = flux + noise

        # Re-normalize to prevent Quantum Angle Wrapping
        flux_mean = noisy_flux.mean(dim=-1, keepdim=True)
        flux_std = noisy_flux.std(dim=-1, keepdim=True) + 1e-6
        item['flux'] = (noisy_flux - flux_mean) / flux_std

        return item

# ---------------------------------------------------------------------------
# EVALUATION & TRAINING LOOPS
# ---------------------------------------------------------------------------
def evaluate_with_noise(model, test_loader, device, sigma):
    """Standard evaluation loop for the final robustness testing."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            flux = batch['flux'].to(device)
            labels = batch['label'].to(device)

            if sigma > 0.0:
                noise = torch.randn_like(flux) * sigma * flux.std(dim=-1, keepdim=True)
                flux = flux + noise

                flux_mean = flux.mean(dim=-1, keepdim=True)
                flux_std = flux.std(dim=-1, keepdim=True) + 1e-6
                flux = (flux - flux_mean) / flux_std

            logits = model(flux)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_and_get_model(ModelClass, model_name, train_loader, val_loader, data_config, trainable_filter, run_dir):
    print(f"\n{'=' * 50}\nTraining {model_name} on Clean Data\n{'=' * 50}")

    model = ModelClass(seq_len=SEQ_LEN, trainable_filter=trainable_filter, num_classes=NUM_CLASSES)
    train_cfg = TrainingConfig(epochs=300, lr=1e-3)

    model_dir = run_dir / model_name

    # The trainer now handles its own internal directories because we pass base_run_dir
    trainer = SDSSPerformanceTrainer(model, train_cfg, run_name=model_name, base_run_dir=str(model_dir))

    # We only need to hijack the resume file so it stays in our Master folder
    trainer.resume_file = str(run_dir / f"resume_{model_name}.pt")

    trainer.train(
        train_loader=train_loader, val_loader=val_loader,
        epochs=train_cfg.epochs, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    best_path = os.path.join(trainer.checkpoint_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_path, map_location=trainer.device))

    return model, trainer.device

def main(base_run_dir, seed=0, head_mode='heavy'):
    qv.LIGHT_HEAD_DIAGNOSTIC = (head_mode == 'light')
    run_dir = Path(base_run_dir) / EXP_NAME / head_mode / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SPLIT_SEED)  # identical data split every run

    print("Loading Pristine Dataset...")
    data_config = SDSSDataConfig(
        parquet_path=str(PROJECT_ROOT / "dataset" / "EXP2_HIGH_SNR_CLEAN_K3_K5.parquet"),
        use_augmentation=False,
        batch_size=64,
        num_workers=0,
        scalar_cols=[],
        train_size=0.50,
        val_size=0.15,
        test_size=0.35
    )
    dm = SDSSDataModule(data_config)
    dm.prepare_data()

    set_seed(seed) # vary only model init

    # ---------------------------------------------------------
    # ADVERSARIAL INJECTION
    # Wrap ONLY the training dataset with dynamic noise.
    # Validation and Test sets remain perfectly clean.
    # ---------------------------------------------------------
    adv_train_ds = AdversarialNoiseWrapper(InMemory(dm.train_ds), min_sigma=0.1, max_sigma=0.6)

    # Create the new adversarial dataloader
    adv_train_loader = DataLoader(
        adv_train_ds,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers
    )

    val_loader = dm.get_loader(dm.val_ds)
    test_loader = dm.get_loader(dm.test_ds)

    models = {}
    models['Frozen_Quanv'], dev = train_and_get_model(QuanvClassifier, "ADV_Frozen_Quanv", adv_train_loader, val_loader,
                                                      data_config, False, run_dir)
    models['Frozen_CNN'], _ = train_and_get_model(ClassicalConvClassifier, "ADV_Frozen_CNN", adv_train_loader,
                                                  val_loader, data_config, False, run_dir)
    models['Trainable_Quanv'], _ = train_and_get_model(QuanvClassifier, "ADV_Trainable_Quanv", adv_train_loader,
                                                       val_loader, data_config, True, run_dir)
    models['Trainable_CNN'], _ = train_and_get_model(ClassicalConvClassifier, "ADV_Trainable_CNN", adv_train_loader,
                                                     val_loader, data_config, True, run_dir)

    print(f"\n{'=' * 50}\nStarting Adversarial Noise Evaluation\n{'=' * 50}")
    results = {name: [] for name in models.keys()}

    for sigma in NOISE_LEVELS:
        print(f"\nTesting with Noise Level: sigma={sigma}")
        for name, model in models.items():
            acc = evaluate_with_noise(model, test_loader, dev, sigma)
            results[name].append(acc)
            print(f"  -> {name}: {acc:.4f}")

    jq, jc = run_jacobian_probe(models['Trainable_Quanv'], models['Trainable_CNN'], test_loader)
    results['_jac_quanv'] = jq
    results['_jac_cnn'] = jc
    results['_sigmas'] = NOISE_LEVELS

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=["heavy", "light"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    for s in args.seeds:
        print(f"\n########## {EXP_NAME} | head={args.head} | seed={s} ##########")
        main(args.run_dir, s, args.head)