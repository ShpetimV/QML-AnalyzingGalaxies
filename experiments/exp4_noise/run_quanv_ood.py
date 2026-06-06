import sys
import os
import random
from pathlib import Path
import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
EXP_NAME = "ood"
SPLIT_SEED = 42

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
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.autograd.functional import jacobian
from scipy.stats import wilcoxon
import torch.nn.functional as F

from src.param_config import SDSSDataConfig, TrainingConfig
from src.sdss_dataloader import SDSSDataModule
from src.training.trainer import SDSSPerformanceTrainer
from src.models.quanvolution import QuanvClassifier, ClassicalConvClassifier
import src.models.quanvolution as qv

# --- CONFIGURATION ---
NUM_CLASSES = 2
SEQ_LEN = 1024  # The models will pool down to this

# Sigma values from 0.0 to 4.0 in increments of 0.1
NOISE_LEVELS = [x / 10.0 for x in range(0, 41)]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# Create a master timestamped folder for THIS specific run
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_RUN_DIR = PROJECT_ROOT / "results_exp2_noise" / "experiment4_quanv_noise_robustness" /  "ood_generalization" / f"Run_{TIMESTAMP}"

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
    return float(np.median(vals)), float(np.subtract(*np.percentile(vals, [75, 25])))  # median, IQR                         # same patches used for BOTH models

def run_jacobian_probe(quanv_model, cnn_model, test_loader, n=300):
    quanv_model.eval().cpu(); cnn_model.eval().cpu()
    patches = sample_test_patches(test_loader, SEQ_LEN, 4, 2, n=n)
    qm, qs = mean_jac_norm(make_quanv_fn(quanv_model.quanv), patches)
    cm, cs = mean_jac_norm(make_classical_fn(cnn_model), patches)
    print(f"[Jacobian] Quanv ||J||_F = {qm:.4f} ± {qs:.4f}")
    print(f"[Jacobian] CNN   ||J||_F = {cm:.4f} ± {cs:.4f}")
    print(f"[Jacobian] ratio Quanv/CNN = {qm/cm:.3f}  (<1 => quantum map is smoother)")
    return qm, cm

def evaluate_with_noise(model, test_loader, device, sigma):
    """Custom evaluation loop that injects mathematical Gaussian noise."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            flux = batch['flux'].to(device)
            labels = batch['label'].to(device)

            if sigma > 0.0:
                # 1. Add the noise
                noise = torch.randn_like(flux) * sigma * flux.std(dim=-1, keepdim=True)
                flux = flux + noise

                # 2. RE-NORMALIZE to prevent Quantum Angle Wrapping
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
    set_seed(SPLIT_SEED) # identical data split every run

    print("Loading Pristine Dataset...")
    data_config = SDSSDataConfig(
        parquet_path=str(PROJECT_ROOT / "dataset" / "EXP2_HIGH_SNR_CLEAN_K3_K5.parquet"),
        use_augmentation=False,
        batch_size=64,
        num_workers=0,
        scalar_cols=[],
        # THE FEW-SHOT CONFIGURATION:
        train_size=0.50,  # 5000 training samples
        val_size=0.15,
        test_size=0.35
    )
    dm = SDSSDataModule(data_config)
    dm.prepare_data()

    set_seed(seed) # vary only model init

    train_loader = dm.get_loader(dm.train_ds, use_sampler=True)
    val_loader = dm.get_loader(dm.val_ds)
    test_loader = dm.get_loader(dm.test_ds)

    models = {}
    models['Frozen_Quanv'], dev = train_and_get_model(QuanvClassifier, "OOD_Frozen_Quanv", train_loader, val_loader,
                                                      data_config, False, run_dir)
    models['Frozen_CNN'], _ = train_and_get_model(ClassicalConvClassifier, "OOD_Frozen_CNN", train_loader, val_loader,
                                                  data_config, False, run_dir)
    models['Trainable_Quanv'], _ = train_and_get_model(QuanvClassifier, "OOD_Trainable_Quanv", train_loader, val_loader,
                                                       data_config, True, run_dir)
    models['Trainable_CNN'], _ = train_and_get_model(ClassicalConvClassifier, "OOD_Trainable_CNN", train_loader,
                                                     val_loader, data_config, True, run_dir)

    print(f"\n{'=' * 50}\nStarting Out-of-Distribution Noise Evaluation\n{'=' * 50}")
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