import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
import random

# 1. Load parquet
print("Loading parquet...")
df = pd.read_parquet("../dataset/sdss_merged_full.parquet")
print(f"Loaded {len(df):,} rows")

# 2. Pick 10 random rows
sample = df.sample(10).reset_index(drop=True)

CLASS_COLORS = {
    'STAR':    '#FFD700',
    'GALAXY':  '#00BFFF',
    'QSO':     '#FF6347',
    'UNKNOWN': '#AAAAAA',
}

def plot_spectra(sample, labeled, output_path):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0d0d1a')
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

    for idx, row in sample.iterrows():
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor('#0d0d1a')

        flux_arr = np.array(row['flux'], dtype=float)
        lam_arr  = np.array(row['lambda'], dtype=float)
        flux_smooth = gaussian_filter1d(flux_arr, sigma=2)

        if labeled:
            label = str(row['class']).strip() if pd.notna(row['class']) else 'UNKNOWN'
            color = CLASS_COLORS.get(label, '#AAAAAA')
            snr   = f"S/N: {row['snMedian']:.1f}" if pd.notna(row['snMedian']) else ""
            z     = f"z={row['Z']:.3f}" if pd.notna(row['Z']) else ""
            title = f"ID: {row['object_id']} — {label}  {z}  {snr}"
        else:
            color = '#AAAAAA'
            title = f"Object #{idx + 1}"

        ax.plot(lam_arr, flux_smooth, color=color, linewidth=0.8, alpha=0.9)
        mask = lam_arr > 3800
        ax.fill_between(lam_arr[mask], flux_smooth[mask], alpha=0.15, color=color)

        ax.set_xlabel('Wavelength (Å)', fontsize=7, color='#888888')
        ax.set_ylabel('Flux', fontsize=7, color='#888888')
        ax.tick_params(colors='#666666', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.set_title(title, fontsize=7, color=color, pad=6, fontweight='bold')

    # Legend only on labeled plot
    if labeled:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=c, linewidth=2, label=k)
                           for k, c in CLASS_COLORS.items()]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                   facecolor='#1a1a2e', edgecolor='#333333',
                   labelcolor='white', fontsize=9, bbox_to_anchor=(0.5, 0.01))

    title_text = 'SDSS Spectral Flux — Labeled' if labeled else 'SDSS Spectral Flux — Unlabeled'
    fig.suptitle(title_text, fontsize=14, color='white', fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved to {output_path}")
    plt.close()

# 3. Save both plots from the same 10 samples
plot_spectra(sample, labeled=False, output_path="./spectra_unlabeled.png")
plot_spectra(sample, labeled=True,  output_path="./spectra_labeled.png")