"""
Spectral Grad-CAM & Saliency for the Quantum Binary Classifier
================================================================
Two complementary analyses:

1. Grad-CAM on the classical CNN extractor's last Conv1d layer
   → Shows which spectral REGIONS the model focuses on (broad patterns)

2. Input saliency (gradient of predicted class w.r.t. raw flux)
   → Shows which individual WAVELENGTH BINS matter most (fine-grained)

Both produce per-class average heatmaps overlaid on the mean spectrum,
so you can see if the model is learning physically meaningful features
(emission lines, continuum breaks, etc.)

Usage (from project root):
    uv run python experiments/gradcam_quantum_binary.py
    uv run python experiments/gradcam_quantum_binary.py --checkpoint path/to/best.pt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.param_config import SDSSDataConfig
from src.sdss_dataloader import SDSSDataModule
from src.models.quantum_model import get_quantum_model


# ---------------------------------------------------------------------------
# Config — must match what you trained with
# ---------------------------------------------------------------------------
CLASS_A = "GALAXY_AGN_BROADLINE"
CLASS_B = "QSO_STARBURST_BROADLINE"

N_QUBITS = 4
N_LAYERS = 4
ENCODING = "amplitude"

CHECKPOINT_PATH = "best_baseline_model.pt"   # default save name from trainer
RESULTS_DIR = "results_quantum_binary"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Binary subset (same as training script)
# ---------------------------------------------------------------------------
class BinarySubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, class_a_idx, class_b_idx):
        mask = (base_dataset.labels == class_a_idx) | (base_dataset.labels == class_b_idx)
        self.indices = torch.where(mask)[0]
        self.base = base_dataset
        old_labels = base_dataset.labels[self.indices]
        self.labels = (old_labels == class_b_idx).long()
        self.is_train = base_dataset.is_train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.base[self.indices[idx].item()]
        sample['label'] = self.labels[idx]
        return sample


# ---------------------------------------------------------------------------
# 1. Grad-CAM for 1D CNN extractor
# ---------------------------------------------------------------------------
class GradCAM1D:
    """
    Grad-CAM adapted for 1D spectra.
    Hooks into the last Conv1d layer of the classical feature extractor.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, flux, scalars, target_class):
        """
        Args:
            flux:         [1, 1, L]
            scalars:      [1, n_scalars]
            target_class: int (0 or 1)
        Returns:
            cam: [L] — Grad-CAM heatmap upsampled to input length
        """
        self.model.eval()
        flux.requires_grad_(True)

        logits = self.model(flux, scalars)
        score = logits[0, target_class]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Global average pooling of gradients → channel weights
        weights = self.gradients.mean(dim=2, keepdim=True)  # [1, C, 1]
        cam = (weights * self.activations).sum(dim=1)       # [1, L_conv]
        cam = F.relu(cam)                                    # only positive contributions

        # Normalise
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input length
        cam = F.interpolate(cam.unsqueeze(0), size=flux.shape[-1],
                            mode='linear', align_corners=False)
        return cam.squeeze().detach().numpy()


# ---------------------------------------------------------------------------
# 2. Input saliency
# ---------------------------------------------------------------------------
def compute_saliency(model, flux, scalars, target_class):
    """
    Raw input gradient saliency.
    Returns |d(score) / d(flux)| at each wavelength bin.
    """
    model.eval()
    flux = flux.clone().detach().requires_grad_(True)

    logits = model(flux, scalars)
    score = logits[0, target_class]

    model.zero_grad()
    score.backward()

    saliency = flux.grad.abs().squeeze().detach().numpy()   # [L]
    return saliency


# ---------------------------------------------------------------------------
# Run analysis over test set
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max samples to process (Grad-CAM is sequential)")
    args = parser.parse_args()

    device = torch.device("cpu")
    class_names = [CLASS_A, CLASS_B]

    # --- Load data ---
    print("Loading data...")
    data_config = SDSSDataConfig(num_workers=0)
    dm = SDSSDataModule(data_config)
    dm.prepare_data()

    all_classes = list(dm.classes)
    idx_a = all_classes.index(CLASS_A)
    idx_b = all_classes.index(CLASS_B)

    test_ds = BinarySubset(dm.test_ds, idx_a, idx_b)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"Test set: {len(test_ds)} samples  ({CLASS_A} vs {CLASS_B})")

    # --- Load model ---
    print(f"Loading model from {args.checkpoint}...")
    model = get_quantum_model(
        encoding=ENCODING,
        num_classes=2,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_scalars=len(data_config.scalar_cols),
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- Find the last Conv1d layer in the extractor ---
    last_conv = None
    for module in model.extractor.backbone.modules():
        if isinstance(module, torch.nn.Conv1d):
            last_conv = module
    if last_conv is None:
        print("ERROR: No Conv1d layer found in extractor")
        return

    print(f"Grad-CAM target layer: {last_conv}")
    gradcam = GradCAM1D(model, last_conv)

    # --- Accumulate heatmaps per class ---
    cam_accum  = {0: [], 1: []}
    sal_accum  = {0: [], 1: []}
    flux_accum = {0: [], 1: []}
    n_processed = 0

    print(f"Processing up to {args.max_samples} samples...")
    for batch in test_loader:
        if n_processed >= args.max_samples:
            break

        flux = batch['flux'].to(device)       # [1, 1, L]
        scalars = batch['scalars'].to(device)  # [1, 6]
        label = batch['label'].item()

        # Grad-CAM for predicted class
        pred = model(flux, scalars).argmax(1).item()

        cam = gradcam(flux.clone(), scalars, target_class=pred)
        sal = compute_saliency(model, flux.clone(), scalars, target_class=pred)

        cam_accum[label].append(cam)
        sal_accum[label].append(sal)
        flux_accum[label].append(flux.squeeze().detach().numpy())

        n_processed += 1
        if n_processed % 50 == 0:
            print(f"  {n_processed} / {args.max_samples}")

    print(f"Processed: {len(cam_accum[0])} {CLASS_A}, {len(cam_accum[1])} {CLASS_B}")

    # --- Compute per-class averages ---
    L = len(cam_accum[0][0]) if cam_accum[0] else len(cam_accum[1][0])
    pixel_axis = np.arange(L)

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    fig.suptitle(f"Spectral attention — {ENCODING} quantum model ({N_QUBITS}q, {N_LAYERS}L)",
                 fontsize=14, y=0.98)

    for cls_idx, cls_name in enumerate(class_names):
        if not cam_accum[cls_idx]:
            continue

        mean_flux = np.mean(flux_accum[cls_idx], axis=0)
        mean_cam  = np.mean(cam_accum[cls_idx], axis=0)
        mean_sal  = np.mean(sal_accum[cls_idx], axis=0)

        # Smooth saliency for readability
        from scipy.ndimage import gaussian_filter1d
        mean_sal_smooth = gaussian_filter1d(mean_sal, sigma=10)

        # Normalise for display
        mean_sal_smooth = mean_sal_smooth / (mean_sal_smooth.max() + 1e-8)

        # --- Grad-CAM plot ---
        ax = axes[cls_idx * 2]
        ax.plot(pixel_axis, mean_flux, color='#888888', linewidth=0.5, alpha=0.6, label='Mean spectrum')
        ax.fill_between(pixel_axis, mean_flux.min(), mean_flux.max(),
                        where=mean_cam > (mean_cam.max() * 0.75),
                        alpha=0.4, color='#E24B4A', label='Grad-CAM (>0.3)')
        ax2 = ax.twinx()
        ax2.plot(pixel_axis, mean_cam, color='#E24B4A', linewidth=1.2, alpha=0.8)
        ax2.set_ylabel('Grad-CAM', color='#E24B4A')
        ax2.set_ylim(0, 1.1)
        ax.set_ylabel('Flux')
        ax.set_title(f'{cls_name} — Grad-CAM (CNN extractor attention)', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

        # --- Saliency plot ---
        ax = axes[cls_idx * 2 + 1]
        ax.plot(pixel_axis, mean_flux, color='#888888', linewidth=0.5, alpha=0.6, label='Mean spectrum')
        ax.fill_between(pixel_axis, mean_flux.min(), mean_flux.max(),
                        where=mean_sal_smooth > 0.3,
                        alpha=0.4, color='#378ADD', label='Saliency (>0.3)')
        ax2 = ax.twinx()
        ax2.plot(pixel_axis, mean_sal_smooth, color='#378ADD', linewidth=1.2, alpha=0.8)
        ax2.set_ylabel('Saliency', color='#378ADD')
        ax2.set_ylim(0, 1.1)
        ax.set_ylabel('Flux')
        ax.set_title(f'{cls_name} — Input saliency (gradient magnitude)', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Spectral pixel (log-λ)')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"gradcam_saliency_{ENCODING}_{N_QUBITS}q.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved → {path}")

    # --- Also save the top-20 most important spectral regions ---
    print("\nTop spectral regions by Grad-CAM intensity:")
    for cls_idx, cls_name in enumerate(class_names):
        if not cam_accum[cls_idx]:
            continue
        mean_cam = np.mean(cam_accum[cls_idx], axis=0)
        # Find peaks (regions above 0.5 threshold)
        hot_pixels = np.where(mean_cam > 0.5)[0]
        if len(hot_pixels) > 0:
            # Group consecutive pixels into regions
            regions = []
            start = hot_pixels[0]
            for i in range(1, len(hot_pixels)):
                if hot_pixels[i] - hot_pixels[i-1] > 5:
                    regions.append((start, hot_pixels[i-1]))
                    start = hot_pixels[i]
            regions.append((start, hot_pixels[-1]))

            print(f"\n  {cls_name}:")
            for s, e in regions[:10]:
                intensity = mean_cam[s:e+1].mean()
                print(f"    pixels {s:4d}–{e:4d}  (width={e-s+1:3d})  intensity={intensity:.3f}")
        else:
            print(f"\n  {cls_name}: no strong regions found (max cam = {mean_cam.max():.3f})")


if __name__ == "__main__":
    main()