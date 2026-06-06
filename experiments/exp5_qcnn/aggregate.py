"""
aggregate.py  --  turn per-seed worker outputs into the thesis result.

Reads:   <run-dir>/<quadrant>/<model>/seed<S>/robustness.csv  (+ jacobian.csv)
Writes:  <run-dir>/auc_summary.csv      mean+-std robust-AUC per (quadrant, model)
         <run-dir>/wilcoxon.csv         paired QCNN-vs-baseline test per quadrant
         <run-dir>/jacobian_summary.csv mean spectral-norm per (quadrant, model)
         <run-dir>/band_<quadrant>.png  mean+-1sigma decay curves
         <run-dir>/jacobian.png         spectral-norm comparison

robust-AUC = normalized area under the accuracy-vs-sigma curve = mean robust acc.
"""
import argparse, glob, os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from pathlib import Path

trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2.x renamed it

# --- STYLING ---
PALETTE = {"QCNN": "#00A9FF", "CNN_Tiny": "#3B9E8C", "CNN_Huge": "#FF6B35", "QCNN_Hyb": "#9C27B0"}
SLOPE_COLORS = {"better": "#00BFFF", "worse": "#FF6B6B"}


def _dark(ax):
    """Apply a dark theme to the given Matplotlib axis."""
    ax.set_facecolor('#0d0d1a')
    ax.figure.set_facecolor('#0d0d1a')
    for s in ax.spines.values():
        s.set_edgecolor('#333333')
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('white')


def plot_jacobian_slopegraph(jac, quadrants, target, baseline, out_dir):
    """Slopegraph per quadrant comparing target vs baseline Jacobian norms."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for i, q in enumerate(quadrants):
        ax = axes[i]
        tq = jac.get((q, target), {})
        bq = jac.get((q, baseline), {})
        shared = sorted(set(tq) & set(bq))
        if not shared:
            ax.text(0.5, 0.5, "no shared seeds", ha='center', va='center', color='#cccccc')
            ax.set_xticks([0, 1]);
            ax.set_xticklabels([target, baseline])
            _dark(ax)
            continue
        for s in shared:
            qv = tq[s]
            bv = bq[s]
            color = SLOPE_COLORS["better"] if qv < bv else SLOPE_COLORS["worse"]
            ax.plot([0, 1], [qv, bv], '-o', color=color, alpha=0.6)
        ax.plot([0, 1], [np.median([tq[s] for s in shared]), np.median([bq[s] for s in shared])],
                '-s', color='white', lw=2.5, label='median')
        ax.set_xticks([0, 1]);
        ax.set_xticklabels([target, baseline])
        ax.set_title(f"{q}")
        ax.set_ylabel(r"$\|J\|_2$ (lower = smoother)")
        _dark(ax)
        ax.legend(facecolor='#0d0d1a', edgecolor='#333333', labelcolor='white')
    fig.suptitle(f"Jacobian slopegraph: {target} vs {baseline}", color='white')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"jacobian_slope_{target}_vs_{baseline}.png"), dpi=200)
    plt.close(fig)


SIGMA_MAX = 1.5   # integrate/plot only here; beyond this every model sits at chance

def robust_auc(acc, sig):
    sig, acc = np.asarray(sig, float), np.asarray(acc, float)
    m = sig <= SIGMA_MAX
    sig, acc = sig[m], acc[m]
    return float(trapz(acc, sig) / (sig[-1] - sig[0]))


def collect(run_dir):
    """Return curves[(quadrant, model)] -> {seed: (sigmas, acc)} and the same for jac spec."""
    curves = defaultdict(dict)
    jac = defaultdict(dict)
    for path in glob.glob(os.path.join(run_dir, "*", "*", "seed*", "robustness.csv")):
        parts = path.split(os.sep)
        quadrant, model, seed = parts[-4], parts[-3], int(parts[-2].replace("seed", ""))
        df = pd.read_csv(path)
        curves[(quadrant, model)][seed] = (df["Sigma"].values, df["Accuracy"].values)
        jpath = os.path.join(os.path.dirname(path), "jacobian.csv")
        if os.path.exists(jpath):
            jac[(quadrant, model)][seed] = pd.read_csv(jpath)["spec"].mean()
    return curves, jac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", default="QCNN", help="model compared against baselines")
    ap.add_argument("--baselines", nargs="+", default=["CNN_Tiny", "CNN_Huge"])
    a = ap.parse_args()

    curves, jac = collect(a.run_dir)
    if not curves:
        raise SystemExit(f"No robustness.csv found under {a.run_dir}")

    quadrants = sorted({q for (q, _) in curves})
    models = sorted({m for (_, m) in curves})

    # ---- per-(quadrant, model) AUC across seeds -----------------------------
    auc_rows, auc_by = [], {}
    clean_rows, clean_by = [], {}  # clean accuracy at sigma=0
    for (q, m), seeds in curves.items():
        per_seed = {s: robust_auc(acc, sig) for s, (sig, acc) in seeds.items()}
        auc_by[(q, m)] = per_seed
        vals = np.array(list(per_seed.values()))
        auc_rows.append({"quadrant": q, "model": m, "n_seeds": len(vals),
                         "auc_mean": vals.mean(), "auc_std": vals.std()})
        clean_seed = {s: float(acc[np.argmin(np.abs(sig))]) for s, (sig, acc) in seeds.items()}
        clean_by[(q, m)] = clean_seed
        cv = np.array(list(clean_seed.values()))
        clean_rows.append({"quadrant": q, "model": m, "n_seeds": len(cv),
                           "clean_mean": cv.mean(), "clean_std": cv.std()})
    pd.DataFrame(auc_rows).sort_values(["quadrant", "model"]).to_csv(
        os.path.join(a.run_dir, "auc_summary.csv"), index=False)
    pd.DataFrame(clean_rows).sort_values(["quadrant", "model"]).to_csv(
        os.path.join(a.run_dir, "clean_summary.csv"), index=False)

    # ---- paired Wilcoxon: target vs each baseline, per quadrant -------------
    compare_models = [a.target]
    if "QCNN_Hyb" in models and "QCNN_Hyb" not in compare_models:
        compare_models.append("QCNN_Hyb")

    w_rows = []
    for q in quadrants:
        for target_model in compare_models:
            tgt = auc_by.get((q, target_model), {})
            if not tgt:
                continue
            for b in a.baselines:
                if target_model == b:
                    continue  # skip comparing a model with itself
                base = auc_by.get((q, b), {})
                if not base:
                    continue
                shared = sorted(set(tgt) & set(base))
                if len(shared) < 2:
                    continue
                x = np.array([tgt[s] for s in shared])
                y = np.array([base[s] for s in shared])
                try:
                    stat, p = wilcoxon(x, y)
                except ValueError:  # all differences zero
                    stat, p = np.nan, 1.0
                w_rows.append({
                    "quadrant": q,
                    "comparison": f"{target_model} vs {b}",
                    "n_pairs": len(shared),
                    f"{target_model}_auc": x.mean(),
                    f"{b}_auc": y.mean(),
                    "delta": x.mean() - y.mean(),
                    "W": stat,
                    "p_value": p
                })
    pd.DataFrame(w_rows).to_csv(os.path.join(a.run_dir, "wilcoxon.csv"), index=False)
    print(pd.DataFrame(w_rows).to_string(index=False) if w_rows else "no Wilcoxon pairs")

    # ---- clean-accuracy (sigma=0) Wilcoxon: same structure, different metric ----
    cw_rows = []
    for q in quadrants:
        for target_model in compare_models:
            tgt = clean_by.get((q, target_model), {})
            if not tgt:
                continue
            for b in a.baselines:
                if target_model == b:
                    continue
                base = clean_by.get((q, b), {})
                if not base:
                    continue
                shared = sorted(set(tgt) & set(base))
                if len(shared) < 2:
                    continue
                x = np.array([tgt[s] for s in shared])
                y = np.array([base[s] for s in shared])
                try:
                    stat, p = wilcoxon(x, y)
                except ValueError:
                    stat, p = np.nan, 1.0
                cw_rows.append({"quadrant": q, "comparison": f"{target_model} vs {b}",
                                "n_pairs": len(shared),
                                f"{target_model}_clean": x.mean(), f"{b}_clean": y.mean(),
                                "delta": x.mean() - y.mean(), "W": stat, "p_value": p})
    pd.DataFrame(cw_rows).to_csv(os.path.join(a.run_dir, "clean_wilcoxon.csv"), index=False)
    print(pd.DataFrame(cw_rows).to_string(index=False) if cw_rows else "no clean Wilcoxon pairs")


    # ---- mean +- 1 sigma band plot per quadrant (dark theme) ----------------
    for q in quadrants:
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in models:
            if (q, m) not in curves:
                continue
            seeds = curves[(q, m)]
            sig = next(iter(seeds.values()))[0]
            stack = np.vstack([acc for (_, acc) in seeds.values()])
            mean, std = stack.mean(0), stack.std(0)
            c = PALETTE.get(m, None)
            ax.plot(sig, mean, label=f"{m} (n={len(seeds)})", color=c, lw=2.5)
            ax.fill_between(sig, mean - std, mean + std, color=c, alpha=0.18)
        ax.axhline(1.0 / 4, ls="--", color="#888888", alpha=0.6, label="chance (0.25)")
        ax.set_title(f"Robustness decay (mean ± 1σ): {q}", fontsize=13)
        ax.set_xlabel("Noise level σ", fontsize=11)
        ax.set_ylabel("Test accuracy", fontsize=11)
        ax.set_ylim(0.1, 1.02)
        ax.set_xlim(0, SIGMA_MAX)
        ax.grid(True, color='#333333', linestyle='--', alpha=0.6)
        ax.legend(facecolor='#0d0d1a', edgecolor='#333333', labelcolor='white', loc='upper right')
        _dark(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(a.run_dir, f"band_{q}.png"), dpi=200)
        plt.close(fig)

    # ---- Jacobian spectral-norm comparison (dark theme) ---------------------
    if jac:
        jrows = []
        for (q, m), seeds in jac.items():
            v = np.array(list(seeds.values()))
            jrows.append({"quadrant": q, "model": m, "spec_mean": v.mean(), "spec_std": v.std()})
        jdf = pd.DataFrame(jrows).sort_values(["quadrant", "model"])
        jdf.to_csv(os.path.join(a.run_dir, "jacobian_summary.csv"), index=False)

        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(quadrants))
        width = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            means = [jdf[(jdf.quadrant == q) & (jdf.model == m)].spec_mean.mean() for q in quadrants]
            errs = [jdf[(jdf.quadrant == q) & (jdf.model == m)].spec_std.mean() for q in quadrants]
            ax.bar(x + i * width, means, width, yerr=errs, capsize=4,
                   label=m, color=PALETTE.get(m, None), alpha=0.85, edgecolor='#555555', linewidth=0.5,
                   error_kw=dict(ecolor='#cccccc'))
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(quadrants, rotation=15, ha="right")
        ax.set_ylabel(r"mean $\|J\|_2$  (lower = smoother = more robust)", fontsize=11)
        ax.set_title("Input-output Jacobian spectral norm (mechanism probe)", fontsize=13)
        ax.grid(axis="y", color='#333333', linestyle='--', alpha=0.6)
        ax.legend(facecolor='#0d0d1a', edgecolor='#333333', labelcolor='white')
        _dark(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(a.run_dir, "jacobian.png"), dpi=200)
        plt.close(fig)

        # Slopegraph(s) per baseline for QCNN
        for b in a.baselines:
            if (b not in models) or (a.target not in models):
                continue
            plot_jacobian_slopegraph(jac, quadrants, a.target, b, a.run_dir)

        # Optional: slopegraph(s) for QCNN_Hyb vs baselines
        if "QCNN_Hyb" in models:
            for b in a.baselines:
                if b not in models:
                    continue
                plot_jacobian_slopegraph(jac, quadrants, "QCNN_Hyb", b, a.run_dir)

        # Add this after the existing QCNN_Hyb block
        if "QCNN_Hyb" in models and a.target in models:
            plot_jacobian_slopegraph(jac, quadrants, "QCNN_Hyb", a.target, a.run_dir)
    else:
        print("[warn] No jacobian.csv found; skipping Jacobian plots. Ensure run_matrix_one.py ran to completion.")

    summary_lines = []
    summary_lines.append(f"RUN DIR: {a.run_dir}")
    summary_lines.append(f"MODELS: {', '.join(models)}")
    summary_lines.append(f"QUADRANTS: {', '.join(quadrants)}")
    for q in quadrants:
        summary_lines.append(f"\n=== {q} ===")
        for (qq, m), seeds in curves.items():
            if qq != q:
                continue
            per_seed = [robust_auc(acc, sig) for (sig, acc) in seeds.values()]
            summary_lines.append(f"  {m}: AUC {np.mean(per_seed):.4f} ± {np.std(per_seed):.4f} (n={len(per_seed)})")
    if jac:
        summary_lines.append("\nJacobian (median per model/quadrant):")
        for q in quadrants:
            for m in models:
                if (q, m) not in jac:
                    continue
                vals = np.array(list(jac[(q, m)].values()))
                summary_lines.append(f"  {q} | {m}: median ||J||_2 = {np.median(vals):.4f}")

    summary = "\n".join(summary_lines)
    (Path(a.run_dir) / "AGGREGATE_SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(f"\nWrote summaries + plots to {a.run_dir}")


if __name__ == "__main__":
    main()
