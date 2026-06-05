"""
EXPERIMENT 2 (ablation sweep) -- frozen-PCA head: VQC vs tiny MLP, multi-width.

PCA-bottleneck sweep for the exp3 4-class task.

Compares FrozenBeastVQCPCAClassifier vs FrozenBeastTinyClassicalPCAClassifier
across multiple bottleneck widths. Both use the SAME frozen Beast extractor
and the SAME frozen PCA(128 → n) projection, so the comparison isolates the
trainable head: VQC + Linear readout (quantum) vs Linear→Tanh→Linear (classical).

For n_qubits = n_features = 4, both models have ~40 trainable params.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import traceback

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.param_config import SDSSDataConfig, TrainingConfig
from src.sdss_dataloader import SDSSDataModule
from src.training.trainer import SDSSPerformanceTrainer
from src.models.exp3_models import (
    FrozenBeastVQCPCAClassifier,
    FrozenBeastTinyClassicalPCAClassifier,
)

# =====================================================================
# EDIT THIS BLOCK TO CONFIGURE THE SWEEP
# =====================================================================
MODEL = "quantum"          # "quantum" or "classical"
SEEDS = [42, 43, 44]


# Quantum: n_layers fixed at 5 to match exp3 V3; vary n_qubits
QUANTUM_CONFIGS = [
    {"n_qubits": 2, "n_layers": 5},
    {"n_qubits": 3, "n_layers": 5},
    {"n_qubits": 4, "n_layers": 5},
]

# Classical: n_features matched to n_qubits for shape parity
CLASSICAL_CONFIGS = [
    {"n_features": 2},
    {"n_features": 3},
    {"n_features": 4},
]

CLASSES = [
    "STAR_BROWN_DWARF_L",
    "STAR_M8",
    "GALAXY_STARBURST",
    "GALAXY_STARFORMING",
]
SAMPLES_PER_CLASS = 1000
BEAST_CHECKPOINT = "src/models/trained_models/baseline_cnn_transformer.pt"
RESULTS_ROOT = "results_sweep_exp2_pca"
# =====================================================================


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def subsample_balanced(dataset, n_per_class, num_classes, seed=42):
    rng = np.random.default_rng(seed)
    current_labels = dataset.full_labels[dataset.indices]
    new_indices = []
    for class_idx in range(num_classes):
        class_positions = np.where(current_labels == class_idx)[0]
        if len(class_positions) > n_per_class:
            chosen = rng.choice(class_positions, n_per_class, replace=False)
        else:
            chosen = class_positions
        new_indices.append(dataset.indices[chosen])
    dataset.indices = np.concatenate(new_indices)


def build_model(model_type: str, config: dict):
    if model_type == "quantum":
        return FrozenBeastVQCPCAClassifier(
            checkpoint_path=BEAST_CHECKPOINT, num_classes=len(CLASSES), **config,
        )
    if model_type == "classical":
        return FrozenBeastTinyClassicalPCAClassifier(
            checkpoint_path=BEAST_CHECKPOINT, num_classes=len(CLASSES), **config,
        )
    raise ValueError(model_type)


def cfg_to_str(config: dict) -> str:
    return "_".join(f"{k.replace('n_', '')}{v}" for k, v in config.items())


def evaluate(model, test_loader, device, class_names):
    model.to(device).eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            flux = batch["flux"].to(device)
            lbl = batch["label"]
            logits = model(flux)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(lbl.numpy())
    y_true = np.array(labels)
    y_pred = np.array(preds)

    report = classification_report(
        y_true, y_pred, target_names=class_names,
        digits=4, output_dict=True, zero_division=0,
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "y_true":   y_true.tolist(),
        "y_pred":   y_pred.tolist(),
    }
    for cls in class_names:
        out[f"recall_{cls}"]    = float(report[cls]["recall"])
        out[f"precision_{cls}"] = float(report[cls]["precision"])
        out[f"f1_{cls}"]        = float(report[cls]["f1-score"])
    return out


def load_best_checkpoint(model, run_name: str, device) -> None:
    matches = glob.glob(f"runs/{run_name}_*")
    if not matches:
        return
    latest = max(matches, key=os.path.getctime)
    best = os.path.join(latest, "trained_models", "best_model.pt")
    try:
        model.load_state_dict(torch.load(best, map_location=device))
    except FileNotFoundError:
        pass


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    data_config = SDSSDataConfig(num_workers=0)
    training_config = TrainingConfig()
    epochs = training_config.epochs
    num_classes = len(CLASSES)

    set_seed(42)
    data_module = SDSSDataModule(data_config)
    data_module.prepare_data(classes=CLASSES)

    subsample_balanced(data_module.train_ds, int(SAMPLES_PER_CLASS * data_config.train_size), num_classes)
    subsample_balanced(data_module.val_ds,   int(SAMPLES_PER_CLASS * data_config.val_size),   num_classes)
    subsample_balanced(data_module.test_ds,  int(SAMPLES_PER_CLASS * data_config.test_size),  num_classes)

    train_loader = data_module.get_loader(data_module.train_ds, use_sampler=True)
    val_loader   = data_module.get_loader(data_module.val_ds)
    test_loader  = data_module.get_loader(data_module.test_ds)

    print(f"\n{num_classes}-class task: {CLASSES}")
    for name, ds in [("Train", data_module.train_ds), ("Val", data_module.val_ds), ("Test", data_module.test_ds)]:
        labels = ds.labels
        counts = {c: (labels == i).sum().item() for i, c in enumerate(CLASSES)}
        print(f"  {name} ({len(ds)}): {counts}")

    configs = QUANTUM_CONFIGS if MODEL == "quantum" else CLASSICAL_CONFIGS
    print(f"\nMODEL: {MODEL}\nSEEDS: {SEEDS}\nCONFIGS: {configs}\nEPOCHS: {epochs}")
    print(f"Total runs: {len(configs) * len(SEEDS)}\n")

    results_path = os.path.join(RESULTS_ROOT, f"sweep_{MODEL}.json")
    results = []

    for config in configs:
        cfg_str = cfg_to_str(config)
        for seed in SEEDS:
            run_name = f"SweepExp3PCA_{MODEL}_{cfg_str}_seed{seed}"
            print(f"\n{'='*70}\n  {run_name}\n{'='*70}")

            try:
                set_seed(seed)
                model = build_model(MODEL, config)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  trainable params: {n_params:,}")

                trainer = SDSSPerformanceTrainer(model, training_config, run_name=run_name)

                print("  Fitting PCA on training features...")
                model.fit_pca(train_loader, trainer.device)

                trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    lr=training_config.lr,
                    weight_decay=training_config.weight_decay,
                )

                load_best_checkpoint(model, run_name, trainer.device)
                metrics = evaluate(model, test_loader, trainer.device, CLASSES)

                slim = {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")}
                print(f"  acc={slim['accuracy']:.4f}  macro_f1={slim['macro_f1']:.4f}")
                print(classification_report(
                    metrics["y_true"], metrics["y_pred"],
                    target_names=CLASSES, digits=4, zero_division=0,
                ))

                results.append({
                    "model":  MODEL,
                    "config": cfg_str,
                    **config,
                    "seed":   seed,
                    "params": n_params,
                    **slim,
                })
            except Exception as exc:
                print(f"  [error] {exc}")
                traceback.print_exc()
                results.append({
                    "model":  MODEL,
                    "config": cfg_str,
                    **config,
                    "seed":   seed,
                    "error":  str(exc),
                })

            with open(results_path, "w") as fh:
                json.dump(results, fh, indent=2)

    # ---- Aggregate ----
    print("\n" + "=" * 90)
    print(f"SWEEP SUMMARY  ({MODEL}, {len(SEEDS)} seeds per config)")
    print("=" * 90)
    print(f"{'config':<14} {'params':>8} {'acc μ':>9} {'acc σ':>9} {'f1 μ':>9} {'f1 σ':>9}")
    print("-" * 64)

    summary = []
    for config in configs:
        cfg_str = cfg_to_str(config)
        rows = [r for r in results if r.get("config") == cfg_str and "error" not in r]
        if not rows:
            print(f"{cfg_str:<14} (no successful runs)")
            continue

        accs = np.array([r["accuracy"] for r in rows])
        f1s  = np.array([r["macro_f1"] for r in rows])
        params = rows[0]["params"]

        print(f"{cfg_str:<14} {params:>8} "
              f"{accs.mean():>9.4f} {accs.std():>9.4f} "
              f"{f1s.mean():>9.4f} {f1s.std():>9.4f}")

        entry = {
            "config":        cfg_str,
            **config,
            "params":        params,
            "n_seeds":       len(rows),
            "acc_mean":      float(accs.mean()),
            "acc_std":       float(accs.std()),
            "macro_f1_mean": float(f1s.mean()),
            "macro_f1_std":  float(f1s.std()),
        }
        for cls in CLASSES:
            for metric in ("recall", "precision", "f1"):
                key = f"{metric}_{cls}"
                if key in rows[0]:
                    arr = np.array([r[key] for r in rows])
                    entry[f"{key}_mean"] = float(arr.mean())
                    entry[f"{key}_std"]  = float(arr.std())
        summary.append(entry)

    summary_path = os.path.join(RESULTS_ROOT, f"sweep_{MODEL}_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nPer-run results: {results_path}")
    print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    main()
