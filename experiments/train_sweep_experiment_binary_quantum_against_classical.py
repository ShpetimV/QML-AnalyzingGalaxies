"""
EXPERIMENT 1 (binary) -- CANONICAL sweep: quantum vs classical, multi-seed.

Sweep training script for the binary BROWN_DWARF_L vs STAR_M8 task.
Produces the canonical Experiment-1 numbers (results_sweep_binary/SWEEP_REPORT.md).
Set MODEL = "quantum" or "classical" at the top of the config block, then run once
per setting. Quantum sweeps 5 (qubit, layer) configs; classical sweeps the Tanh
mirror widths. Reports mean +/- std per config across seeds.

Loops over (config, seed) pairs and reports mean ± std per config across seeds.

Results are written to results_sweep_binary/ and runs/Sweep_*.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import traceback

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    classification_report, f1_score, roc_auc_score,
)

from src.param_config import SDSSDataConfig, TrainingConfig
from src.sdss_dataloader import SDSSDataModule
from src.training.trainer import SDSSPerformanceTrainer
from src.models.quantum_model import AngleEncodingClassifier
from src.models.classical_mirror import ClassicalMirrorTanhClassifier

# =====================================================================
# EDIT THIS BLOCK TO CONFIGURE THE SWEEP
# =====================================================================

MODEL = "classical"          # "quantum" or "classical"
SEEDS = [42, 43, 44]       # add/remove seeds here


# (n_qubits, n_layers) — used when MODEL == "quantum"
QUANTUM_CONFIGS = [
    {"n_qubits": 2, "n_layers": 12},
    {"n_qubits": 3, "n_layers": 8},
    {"n_qubits": 4, "n_layers": 6},
    {"n_qubits": 6, "n_layers": 4},
    {"n_qubits": 8, "n_layers": 3},
]

# n_features for the dead-ReLU-fixed classical mirror
CLASSICAL_CONFIGS = [
    {"n_features": 4},
    {"n_features": 8},
]

CLASS_A = "STAR_BROWN_DWARF_L"
CLASS_B = "STAR_M8"
RESULTS_ROOT = "results_sweep_binary"

# =====================================================================


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_model(model_type: str, config: dict, dropout: float):
    if model_type == "quantum":
        return AngleEncodingClassifier(num_classes=2, dropout=dropout, **config)
    if model_type == "classical":
        return ClassicalMirrorTanhClassifier(num_classes=2, dropout=dropout, **config)
    raise ValueError(f"Unknown MODEL: {model_type}")


def cfg_to_str(config: dict) -> str:
    return "_".join(f"{k.replace('n_', '')}{v}" for k, v in config.items())


def evaluate(model, test_loader, device, class_names) -> dict:
    model.to(device).eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            flux = batch['flux'].to(device)
            labels = batch['label'].to(device)
            logits = model(flux)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    report = classification_report(
        y_true, y_pred, target_names=class_names,
        digits=4, output_dict=True, zero_division=0,
    )

    out = {
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "macro_f1":      float(f1_score(y_true, y_pred, average='macro')),
        "roc_auc":       float(roc_auc_score(y_true, y_probs)),
        "avg_precision": float(average_precision_score(y_true, y_probs)),
        "y_true":        y_true.tolist(),
        "y_pred":        y_pred.tolist(),
    }
    for cls in class_names:
        out[f"recall_{cls}"]    = float(report[cls]["recall"])
        out[f"precision_{cls}"] = float(report[cls]["precision"])
        out[f"f1_{cls}"]        = float(report[cls]["f1-score"])
    return out


def load_best_checkpoint(model, run_name: str, device) -> None:
    matches = glob.glob(f"runs/{run_name}_*")
    if not matches:
        print(f"  [warn] no run dir matching runs/{run_name}_*; using current weights")
        return
    latest = max(matches, key=os.path.getctime)
    best = os.path.join(latest, "trained_models", "best_model.pt")
    try:
        model.load_state_dict(torch.load(best, map_location=device))
    except FileNotFoundError:
        print(f"  [warn] {best} not found; using current weights")


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    data_config = SDSSDataConfig(num_workers=0)
    training_config = TrainingConfig()
    epochs = training_config.epochs

    # Construct dataset once with a fixed split-seed so all runs see identical splits
    set_seed(42)
    data_module = SDSSDataModule(data_config)
    data_module.prepare_data(classes=[CLASS_A, CLASS_B])
    binary_classes = list(data_module.classes)

    train_loader = data_module.get_loader(data_module.train_ds, use_sampler=True)
    val_loader   = data_module.get_loader(data_module.val_ds)
    test_loader  = data_module.get_loader(data_module.test_ds)

    print(f"\nBinary task: {CLASS_A} (0) vs {CLASS_B} (1)")
    for name, ds in [("Train", data_module.train_ds), ("Val", data_module.val_ds), ("Test", data_module.test_ds)]:
        labels = ds.labels
        print(f"  {name}: {CLASS_A}={(labels==0).sum().item()}  {CLASS_B}={(labels==1).sum().item()}")

    configs = QUANTUM_CONFIGS if MODEL == "quantum" else CLASSICAL_CONFIGS
    print(f"\nMODEL: {MODEL}")
    print(f"SEEDS: {SEEDS}")
    print(f"CONFIGS: {configs}")
    print(f"EPOCHS: {epochs}")
    print(f"Total runs: {len(configs) * len(SEEDS)}\n")

    results_path = os.path.join(RESULTS_ROOT, f"sweep_{MODEL}.json")
    results = []

    for config in configs:
        cfg_str = cfg_to_str(config)
        for seed in SEEDS:
            run_name = f"Sweep_{MODEL}_{cfg_str}_seed{seed}"
            print(f"\n{'='*70}\n  {run_name}\n{'='*70}")

            try:
                set_seed(seed)
                model = build_model(MODEL, config, training_config.dropout)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  trainable params: {n_params:,}")

                trainer = SDSSPerformanceTrainer(model, training_config, run_name=run_name)

                trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    lr=training_config.lr,
                    weight_decay=training_config.weight_decay,
                )

                load_best_checkpoint(model, run_name, trainer.device)
                metrics = evaluate(model, test_loader, trainer.device, binary_classes)

                # Drop the verbose label arrays from the per-run summary
                slim = {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")}
                print(f"  acc={slim['accuracy']:.4f}  macro_f1={slim['macro_f1']:.4f}  "
                      f"auc={slim['roc_auc']:.4f}  ap={slim['avg_precision']:.4f}")

                # Brief per-class report so the bias direction is visible
                print(classification_report(
                    metrics["y_true"], metrics["y_pred"],
                    target_names=binary_classes, digits=4, zero_division=0,
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
                print(f"  [error] run failed: {exc}")
                traceback.print_exc()
                results.append({
                    "model":  MODEL,
                    "config": cfg_str,
                    **config,
                    "seed":   seed,
                    "error":  str(exc),
                })

            # Persist after every run so a crash later doesn't lose finished work
            with open(results_path, "w") as fh:
                json.dump(results, fh, indent=2)

    # ---------------------------------------------------------------
    # Aggregate across seeds, per config
    # ---------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"SWEEP SUMMARY  ({MODEL}, {len(SEEDS)} seeds per config)")
    print("=" * 90)
    header = f"{'config':<14} {'params':>8} {'acc μ':>9} {'acc σ':>9} {'f1 μ':>9} {'f1 σ':>9} {'auc μ':>9} {'auc σ':>9}"
    print(header)
    print("-" * len(header))

    summary = []
    for config in configs:
        cfg_str = cfg_to_str(config)
        rows = [r for r in results if r.get("config") == cfg_str and "error" not in r]
        if not rows:
            print(f"{cfg_str:<14} (no successful runs)")
            continue

        accs = np.array([r["accuracy"] for r in rows])
        f1s  = np.array([r["macro_f1"] for r in rows])
        aucs = np.array([r["roc_auc"]  for r in rows])
        params = rows[0]["params"]

        print(f"{cfg_str:<14} {params:>8} "
              f"{accs.mean():>9.4f} {accs.std():>9.4f} "
              f"{f1s.mean():>9.4f} {f1s.std():>9.4f} "
              f"{aucs.mean():>9.4f} {aucs.std():>9.4f}")

        entry = {
            "config":         cfg_str,
            **config,
            "params":         params,
            "n_seeds":        len(rows),
            "acc_mean":       float(accs.mean()),
            "acc_std":        float(accs.std()),
            "macro_f1_mean":  float(f1s.mean()),
            "macro_f1_std":   float(f1s.std()),
            "roc_auc_mean":   float(aucs.mean()),
            "roc_auc_std":    float(aucs.std()),
        }
        for cls in binary_classes:
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
