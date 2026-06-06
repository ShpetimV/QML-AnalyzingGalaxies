"""
Run modes:
    python run_exp_rigorous.py              # real SDSS data (edit load_data)
    python run_exp_rigorous.py --synthetic  # standalone smoke test, no repo deps
    python run_exp_rigorous.py --quick      # fewer seeds / sizes (fast)

in Thesis:
    n-feature sweep: for f in 4 6 8 12 16; do uv run python -m experiments.run_exp1_fewshot_new --features $f --out experiment_results/experiment3_fewshot/results_sweep/f$f; done
    c-value sweep: for c in 0.1 0.25 0.5 0.75 1.0 1.5; do uv run python -m experiments.run_exp1_fewshot_new --features 12 --bandwidth $c --out experiment_results/experiment3_fewshot/results_bw/c$c; done
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

# Local engine (this file lives next to qsvm.py for the smoke test; in the repo
# it is `from src.models.qsvm import ...`).
try:
    from qsvm import QuantumSVM, geometric_difference
except ImportError:
    from src.models.qsvm import QuantumSVM, geometric_difference  # noqa


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
N_FEATURES = 6          # qubits. Keep modest: large qubit counts concentrate kernels.
REPS = 1                # feature-map repetitions (data re-uploading depth)
BANDWIDTH = 1.0          # encoding bandwidth: data scale before encoding (Shaydulin & Wild 2022)
SAMPLE_SIZES = [2, 5, 10, 20, 50, 100, 200]
N_SEEDS = 8             # training-subset resamples per (method, n)
TEST_PER_CLASS = 250
PRESCREEN_N_PER_CLASS = 25   # subset size for the geometric-difference pre-screen
REG_SWEEP = [1e-6, 1e-8, 1e-10]  # report g across regularisations (it is a choice)

# Classes used when loading the real SDSS data (edit to taste).
TARGET_CLASSES = ["GALAXY_STARBURST", "GALAXY_STARFORMING",
                  "QSO_STARBURST_BROADLINE", "QSO_STARFORMING"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data_synthetic(n_features, n_classes=4, seed=0):
    """Self-contained data so the harness runs without the SDSS repo."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=4000, n_features=n_features, n_informative=n_features,
        n_redundant=0, n_classes=n_classes, n_clusters_per_class=1,
        class_sep=1.0, flip_y=0.05, random_state=seed,
    )
    cut = 2000
    return X[:cut], y[:cut], X[cut:], y[cut:]


def _extract_pooled(loader, n_bins):
    """Pull every batch, mean-pool the 4096-pixel flux down to n_bins pixels."""
    import torch.nn.functional as F
    xs, ys = [], []
    for batch in loader:
        flux = batch["flux"]                       # [B, 1, 4096]
        pooled = F.adaptive_avg_pool1d(flux, output_size=n_bins)
        xs.append(pooled.squeeze(1).numpy())
        ys.extend(batch["label"].numpy())
    return np.vstack(xs), np.array(ys)


def load_data_sdss(n_features):
    """
    Raw-signal track: pool the spectra to `n_features` bins (one per qubit) and
    return the full train/test arrays UNSCALED. The harness does its own
    balanced subsetting + StandardScaler per fold, so we only hand back raw rows.
    """
    import torch
    from src.param_config import SDSSDataConfig
    from src.sdss_dataloader import SDSSDataModule

    cfg = SDSSDataConfig(use_augmentation=False, num_workers=0)
    dm = SDSSDataModule(cfg)
    dm.prepare_data(classes=TARGET_CLASSES)

    # Fix the train loader's shuffle so the extracted row order is reproducible
    # across runs -- otherwise a given seed would pick different samples each run.
    torch.manual_seed(0)
    Xtr, ytr = _extract_pooled(dm.get_loader(dm.train_ds, use_sampler=False), n_features)
    Xte, yte = _extract_pooled(dm.get_loader(dm.test_ds,  use_sampler=False), n_features)
    return Xtr, ytr.astype(int), Xte, yte.astype(int)


def balanced_subset(X, y, n_per_class, rng):
    """Random balanced subset; returns indices. n_per_class is honoured exactly."""
    classes = np.unique(y)
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        take = min(n_per_class, len(c_idx))
        idx.append(rng.choice(c_idx, size=take, replace=False))
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def tuned_classical_svm(X, y):
    """RBF SVM with a real (C, gamma) grid search -- the fair baseline."""
    n_per_class = np.bincount(y).min()
    cv = max(2, min(3, n_per_class))
    grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]}
    try:
        gs = GridSearchCV(SVC(kernel="rbf"), grid,
                          cv=StratifiedKFold(cv, shuffle=True, random_state=0),
                          n_jobs=-1)
        gs.fit(X, y)
        return gs.best_estimator_
    except ValueError:
        clf = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X, y)
        return clf


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def mean_ci(vals, conf=0.95):
    vals = np.asarray(vals, float)
    m = vals.mean()
    if len(vals) < 2:
        return m, m, m
    sem = stats.sem(vals)
    h = sem * stats.t.ppf((1 + conf) / 2.0, len(vals) - 1)
    return m, m - h, m + h


def paired_pvalues(q_vals, c_vals):
    """Paired tests on per-seed (quantum - classical) differences."""
    q, c = np.asarray(q_vals), np.asarray(c_vals)
    out = {}
    try:
        out["ttest_rel_p"] = float(stats.ttest_rel(q, c).pvalue)
    except Exception:
        out["ttest_rel_p"] = float("nan")
    try:
        if np.allclose(q, c):
            out["wilcoxon_p"] = 1.0
        else:
            out["wilcoxon_p"] = float(stats.wilcoxon(q, c).pvalue)
    except Exception:
        out["wilcoxon_p"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def prescreen(X_train_full, y_train_full):
    """Geometric difference g(K_C || K_Q) for both quantum kernels."""
    rng = np.random.default_rng(0)
    idx = balanced_subset(X_train_full, y_train_full, PRESCREEN_N_PER_CLASS, rng)
    Xs = StandardScaler().fit_transform(X_train_full[idx])
    Kc = rbf_kernel(Xs, Xs, gamma=1.0 / Xs.shape[1])
    report = {}
    for kt in ("fidelity", "projected"):
        q = QuantumSVM(n_features=N_FEATURES, kernel_type=kt, reps=REPS, bandwidth=BANDWIDTH)
        Kq = q.train_gram(Xs)
        gs = {f"reg={r:.0e}": geometric_difference(Kc, Kq, reg=r) for r in REG_SWEEP}
        offdiag = Kq[np.triu_indices(len(Kq), 1)]
        report[kt] = {"g": gs, "offdiag_std": float(offdiag.std())}
    return report


def run(use_synthetic, quick, results_dir, run_dense_net=False):
    global N_SEEDS, SAMPLE_SIZES
    if quick:
        N_SEEDS = 3
        SAMPLE_SIZES = [2, 10, 50]

    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(results_dir, f"run_{ts}.log")
    log_lines = []
    _log_fh = open(log_path, "w", buffering=1)  # line-buffered: survives Ctrl-C

    def log(msg):
        print(msg)
        log_lines.append(msg)
        _log_fh.write(msg + "\n")
        _log_fh.flush()

    log(f"Writing results to: {results_dir}")
    log(f"  live log:        {log_path}")
    log(f"  partial results: {os.path.join(results_dir, f'results_{ts}.npz')} "
        f"(rewritten after every sample size)")

    # ---- data ----
    if use_synthetic:
        Xtr_full, ytr_full, Xte_full, yte_full = load_data_synthetic(N_FEATURES)
        log("Using SYNTHETIC data (smoke test).")
    else:
        Xtr_full, ytr_full, Xte_full, yte_full = load_data_sdss(N_FEATURES)
        log("Using SDSS data.")

    # balanced test set
    rng = np.random.default_rng(0)
    te_idx = balanced_subset(Xte_full, yte_full, TEST_PER_CLASS, rng)
    Xte_full, yte_full = Xte_full[te_idx], yte_full[te_idx]
    n_classes = len(np.unique(ytr_full))
    smallest_class = np.bincount(ytr_full).min()
    log(f"Classes={n_classes}  features/qubits={N_FEATURES}  "
        f"smallest train class={smallest_class}  test={len(yte_full)}")

    # ---- PHASE 1: pre-screen ----
    log("\n--- PHASE 1: geometric-difference pre-screen ---")
    pre = prescreen(Xtr_full, ytr_full)
    for kt, info in pre.items():
        gtxt = ", ".join(f"{k}:{v:.2f}" for k, v in info["g"].items())
        log(f"  {kt:10s}  g(C||Q) = [{gtxt}]   off-diag std={info['offdiag_std']:.4f}")
    log("  (g~1 => no advantage possible; off-diag std~0 => kernel concentrated)")

    # ---- PHASE 2: scaling sweep ----
    log("\n--- PHASE 2: multi-seed scaling sweep ---")
    methods = ["classical_rbf", "qsvm_fidelity", "qsvm_projected"]
    if run_dense_net:
        methods.insert(0, "dense_net")
    # results[method] = list over SAMPLE_SIZES of arrays(seed,)
    results = {m: [] for m in methods}

    sizes = [n for n in SAMPLE_SIZES if n <= smallest_class] or [smallest_class]

    for n in sizes:
        log(f"\nn = {n} samples/class")
        per_seed = {m: [] for m in methods}
        for s in range(N_SEEDS):
            rng = np.random.default_rng(1000 + s)
            tr_idx = balanced_subset(Xtr_full, ytr_full, n, rng)
            Xn, yn = Xtr_full[tr_idx], ytr_full[tr_idx]
            scaler = StandardScaler().fit(Xn)
            Xn_s = scaler.transform(Xn)
            Xte_s = scaler.transform(Xte_full)

            csvm = tuned_classical_svm(Xn_s, yn)
            per_seed["classical_rbf"].append(accuracy_score(yte_full, csvm.predict(Xte_s)))

            qf = QuantumSVM(n_features=N_FEATURES, kernel_type="fidelity", reps=REPS, bandwidth=BANDWIDTH)
            qf.fit(Xn_s, yn)
            per_seed["qsvm_fidelity"].append(accuracy_score(yte_full, qf.predict(Xte_s)))

            # Projected kernel = bandwidth-TUNED RBF on quantum-projected features
            # (same grid search the classical baseline gets -> fair comparison).
            qp = QuantumSVM(n_features=N_FEATURES, kernel_type="projected", reps=REPS, bandwidth=BANDWIDTH)
            Ptr = qp.projected_features(Xn_s)
            Pte = qp.projected_features(Xte_s)
            pclf = tuned_classical_svm(Ptr, yn)
            per_seed["qsvm_projected"].append(accuracy_score(yte_full, pclf.predict(Pte)))

            if run_dense_net:
                per_seed["dense_net"].append(_dense_net_acc(Xn_s, yn, Xte_s, yte_full,
                                                            N_FEATURES, n_classes, s))

        for m in methods:
            results[m].append(np.array(per_seed[m]))
            mean, lo, hi = mean_ci(per_seed[m])
            log(f"  {m:15s} {mean:.4f}  (95% CI [{lo:.4f}, {hi:.4f}])")

        # paired tests + pre-registered advantage check vs best classical
        best_classical = np.maximum(np.array(per_seed["classical_rbf"]),
                                    np.array(per_seed.get("dense_net",
                                                          per_seed["classical_rbf"])))
        _, _, c_hi = mean_ci(best_classical)
        for qk in ("qsvm_fidelity", "qsvm_projected"):
            pv = paired_pvalues(per_seed[qk], best_classical)
            _, q_lo, _ = mean_ci(per_seed[qk])
            advantage = q_lo > c_hi
            log(f"    {qk} vs best-classical: paired t p={pv['ttest_rel_p']:.3f}, "
                f"Wilcoxon p={pv['wilcoxon_p']:.3f}, "
                f"PRE-REGISTERED advantage={advantage}")

        # ---- checkpoint after EVERY sample size (survives Ctrl-C / crash) ----
        done = sizes[:len(results[methods[0]])]
        np.savez(os.path.join(results_dir, f"results_{ts}.npz"),
                 sizes=np.array(done),
                 **{m: np.array(results[m]) for m in methods})
        _write_csv(os.path.join(results_dir, f"results_{ts}.csv"),
                   done, results, methods)

    # ---- PHASE 3: final plot ----
    _plot(sizes, results, methods, results_dir, ts)
    log(f"\nSaved results, plot, and log to {results_dir} (stamp {ts})")
    _log_fh.close()
    return os.path.join(results_dir, f"scaling_{ts}.png")


def _write_csv(path, sizes, results, methods):
    """Tidy long-format CSV: one row per (n, method, seed)."""
    with open(path, "w") as f:
        f.write("n_per_class,method,seed,accuracy\n")
        for i, n in enumerate(sizes):
            for m in methods:
                for s, acc in enumerate(results[m][i]):
                    f.write(f"{n},{m},{s},{acc:.6f}\n")


def _dense_net_acc(Xtr, ytr, Xte, yte, in_dim, n_classes, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    from src.models.classical_baselines import SmallDenseNet
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.long, device=dev)
    model = SmallDenseNet(input_dim=in_dim, num_classes=n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    best_state, best_loss, patience, bad = None, 1e9, 30, 0
    model.train()
    for _ in range(300):  # early stopping instead of a fixed 150 epochs
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward(); opt.step()
        if loss.item() < best_loss - 1e-4:
            best_loss, bad, best_state = loss.item(), 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(Xte, dtype=torch.float32, device=dev)).argmax(1).cpu().numpy()
    return accuracy_score(yte, preds)


def _plot(sizes, results, methods, results_dir, ts):
    styles = {
        "dense_net":      ("o", "#FF6347", "Classical Dense Net"),
        "classical_rbf":  ("s", "#FFD700", "Classical SVM (RBF, tuned)"),
        "qsvm_fidelity":  ("*", "#00BFFF", "Quantum SVM (fidelity)"),
        "qsvm_projected": ("D", "#7CFC00", "Quantum SVM (projected/PQK)"),
    }
    plt.figure(figsize=(10, 6))
    for m in methods:
        arr = np.array(results[m])              # (n_sizes, n_seeds)
        means = arr.mean(axis=1)
        cis = np.array([mean_ci(row) for row in arr])
        lo, hi = cis[:, 1], cis[:, 2]
        mk, col, lab = styles[m]
        plt.plot(sizes, means, marker=mk, linestyle="-", color=col, linewidth=2,
                 markersize=9, label=lab)
        plt.fill_between(sizes, lo, hi, color=col, alpha=0.18)
    plt.title("Few-Shot Scaling: Quantum vs Classical (mean +/- 95% CI)",
              fontsize=14, color="white")
    plt.xlabel("Training samples per class", fontsize=12, color="#cccccc")
    plt.ylabel("Test accuracy", fontsize=12, color="#cccccc")
    plt.xscale("log")
    plt.xticks(sizes, [str(s) for s in sizes], color="#cccccc")
    plt.yticks(color="#cccccc")
    ax = plt.gca()
    ax.set_facecolor("#0d0d1a"); plt.gcf().set_facecolor("#0d0d1a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.7)
    plt.legend(facecolor="#0d0d1a", edgecolor="#333333", labelcolor="white")
    plt.tight_layout()
    out = os.path.join(results_dir, f"scaling_{ts}.png")
    plt.savefig(out, dpi=300)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dense-net", action="store_true")
    ap.add_argument("--features", type=int, default=None,
                    help="qubits / pooled feature bins (overrides N_FEATURES)")
    ap.add_argument("--reps", type=int, default=None,
                    help="feature-map repetitions (overrides REPS)")
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="encoding bandwidth c; smaller = less concentration (overrides BANDWIDTH)")
    ap.add_argument("--out", default="results_rigorous")
    args = ap.parse_args()
    if args.features is not None:
        N_FEATURES = args.features
    if args.reps is not None:
        REPS = args.reps
    if args.bandwidth is not None:
        BANDWIDTH = args.bandwidth
    run(args.synthetic, args.quick, args.out, run_dense_net=args.dense_net)