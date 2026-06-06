#!/usr/bin/env python3
"""Aggregate exp2_noise results: robustness bands + Jacobian slopegraphs + paired Wilcoxon.
Reads {run_dir}/{exp}/{head}/seed_*/results.json written by the run scripts."""
import argparse, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless server
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import csv

TRAPZ = np.trapezoid

COLORS = {'Frozen_Quanv': '#00A9FF', 'Frozen_CNN': '#3B9E8C',
          'Trainable_Quanv': '#FF6B35', 'Trainable_CNN': '#9C27B0'}
LS = {'Frozen_Quanv': '-', 'Frozen_CNN': '--', 'Trainable_Quanv': '-', 'Trainable_CNN': '--'}
PAIRS = [('Trainable_Quanv', 'Trainable_CNN'), ('Frozen_Quanv', 'Frozen_CNN')]


def load_results(run_dir):
    run_dir = Path(run_dir)
    data = defaultdict(dict)        # data[(exp, head)][seed] = results dict
    for f in sorted(run_dir.glob("*/*/seed_*/results.json")):
        exp, head = f.parts[-4], f.parts[-3]
        seed = int(re.search(r"seed_(\d+)", f.parts[-2]).group(1))
        with open(f) as fh:
            data[(exp, head)][seed] = json.load(fh)
    return data


def get_sigmas(res):
    if "_sigmas" in res:
        return np.asarray(res["_sigmas"], float)
    n = len(next(v for k, v in res.items() if not k.startswith("_")))
    return np.arange(n) * 0.1       # fallback: assumes 0.1 step from 0

def dump_curves_csv(seeds_dict, sigmas, exp, head, out):
    seeds = sorted(seeds_dict)
    names = [k for k in seeds_dict[seeds[0]] if not k.startswith("_")]
    arrs = {n: np.array([seeds_dict[s][n] for s in seeds]) for n in names}  # (n_seeds, n_sigmas)
    path = out / f"{exp}_{head}_curves.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma"] + [c for n in names for c in (f"{n}_mean", f"{n}_std")])
        for i, sig in enumerate(sigmas):
            row = [f"{sig:.2f}"]
            for n in names:
                row += [f"{arrs[n][:, i].mean():.4f}", f"{arrs[n][:, i].std():.4f}"]
            w.writerow(row)


def robustness_auc(accs, sigmas, window=None):
    accs, sigmas = np.asarray(accs, float), np.asarray(sigmas, float)
    if window is not None:
        m = (sigmas >= window[0]) & (sigmas <= window[1])
        accs, sigmas = accs[m], sigmas[m]
    return TRAPZ(accs, sigmas) / (sigmas[-1] - sigmas[0])


def paired_wilcoxon(a, b):
    """Returns (p_or_None, p_floor, n, note). p_floor = 2/2^n is the min achievable."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    floor = 2.0 / (2 ** n) if n > 0 else 1.0
    if n < 6:
        return None, floor, n, f"n<6 -> significance unreachable (floor={floor:.3f})"
    if np.all(a - b == 0):
        return None, floor, n, "all paired differences are zero"
    try:
        _, p = wilcoxon(a, b)
        return p, floor, n, ""
    except Exception as e:
        return None, floor, n, f"wilcoxon error: {e}"


def _dark(ax):
    ax.set_facecolor('#0d0d1a'); ax.figure.set_facecolor('#0d0d1a')
    for s in ax.spines.values(): s.set_edgecolor('#333333')
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc'); ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('white')


def plot_robustness(seeds_dict, sigmas, exp, head, out):
    seeds = sorted(seeds_dict)
    names = [k for k in seeds_dict[seeds[0]] if not k.startswith("_")]
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in names:
        arr = np.array([seeds_dict[s][name] for s in seeds])   # (n_seeds, n_sigmas)
        mean, std = arr.mean(0), arr.std(0)
        c = COLORS.get(name)
        ax.plot(sigmas, mean, label=name.replace('_', ' '), color=c, ls=LS.get(name, '-'), lw=2)
        ax.fill_between(sigmas, mean - std, mean + std, color=c, alpha=0.18)
    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel("Noise Level (Sigma)"); ax.set_ylabel("Test Accuracy"); ax.set_ylim(0.4, 1.05)
    ax.set_title(f"{exp.upper()} — {head} head (mean ± 1σ over {len(seeds)} seeds)")
    ax.grid(True, color='#333333', ls='--', alpha=0.7)
    _dark(ax)
    ax.legend(facecolor='#0d0d1a', edgecolor='#333333', labelcolor='white')
    fig.tight_layout(); fig.savefig(out / f"{exp}_{head}_robustness.png", dpi=200); plt.close(fig)


def plot_jacobian(seeds_dict, exp, head, out):
    seeds = [s for s in sorted(seeds_dict) if "_jac_quanv" in seeds_dict[s]]
    if not seeds:
        return
    jq = [seeds_dict[s]["_jac_quanv"] for s in seeds]
    jc = [seeds_dict[s]["_jac_cnn"] for s in seeds]
    fig, ax = plt.subplots(figsize=(5, 6))
    for q, c in zip(jq, jc):
        ax.plot([0, 1], [q, c], '-o', color=('#00BFFF' if q < c else '#FF6B6B'), alpha=0.5)
    ax.plot([0, 1], [np.median(jq), np.median(jc)], '-s', color='white', lw=3, label='median')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Quanv', 'CNN'])
    ax.set_ylabel(r'$\|J\|_F$ per seed (lower = smoother)')
    ax.set_title(f"Feature-map sensitivity\n{exp.upper()} — {head} head")
    _dark(ax)
    ax.legend(facecolor='#0d0d1a', edgecolor='#333333', labelcolor='white')
    fig.tight_layout(); fig.savefig(out / f"{exp}_{head}_jacobian.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--auc-window", type=float, nargs=2, default=None,
                    help="restrict AUC to [lo hi], e.g. --auc-window 0 1.5")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    window = tuple(args.auc_window) if args.auc_window else None

    data = load_results(run_dir)
    if not data:
        print(f"No results.json found under {run_dir}"); return
    plots = run_dir / "plots"; plots.mkdir(exist_ok=True)

    lines = []
    for (exp, head), sd in sorted(data.items()):
        seeds = sorted(sd)
        sigmas = get_sigmas(sd[seeds[0]])
        plot_robustness(sd, sigmas, exp, head, plots)
        plot_jacobian(sd, exp, head, plots)

        dump_curves_csv(sd, sigmas, exp, head, plots)

        lines.append(f"\n=== {exp.upper()} | {head} head | {len(seeds)} seeds (auc_window={window}) ===")
        res0 = sd[seeds[0]]
        for qn, cn in PAIRS:
            if qn not in res0 or cn not in res0:
                continue
            q = [robustness_auc(sd[s][qn], sigmas, window) for s in seeds]
            c = [robustness_auc(sd[s][cn], sigmas, window) for s in seeds]
            p, floor, n, note = paired_wilcoxon(q, c)
            ps = "NA" if p is None else f"{p:.4f}"
            lines.append(f"  {qn} vs {cn}: Q={np.mean(q):.4f}±{np.std(q):.4f}  "
                         f"C={np.mean(c):.4f}±{np.std(c):.4f}  Δ={np.mean(q)-np.mean(c):+.4f}  "
                         f"p={ps}  [floor {floor:.3f}] {note}")
        if "_jac_quanv" in res0:
            jq = [sd[s]["_jac_quanv"] for s in seeds]
            jc = [sd[s]["_jac_cnn"] for s in seeds]
            p, floor, n, note = paired_wilcoxon(jq, jc)
            ps = "NA" if p is None else f"{p:.4f}"
            lines.append(f"  Jacobian ||J||_F: Quanv {np.median(jq):.4f}  CNN {np.median(jc):.4f}  "
                         f"(lower=smoother)  p={ps}  [floor {floor:.3f}] {note}")

    summary = "\n".join(lines)
    print(summary)
    (run_dir / "AGGREGATE_SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(f"\nPlots -> {plots}\nSummary -> {run_dir/'AGGREGATE_SUMMARY.txt'}")


if __name__ == "__main__":
    main()