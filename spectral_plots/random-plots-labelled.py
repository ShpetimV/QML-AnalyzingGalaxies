import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
import random
import ast

# 1. Load parquet
print("Loading parquet...")
df = pd.read_parquet("../dataset/sdss_merged_full.parquet")
print(f"Loaded {len(df):,} rows")

# 2. Pick 10 random rows
sample = df.sample(10).reset_index(drop=True)

# 3. Plot
CLASS_COLORS = {
    'STAR':    '#FFD700',
    'GALAXY':  '#00BFFF',
    'QSO':     '#FF6347',
    'UNKNOWN': '#AAAAAA',
}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d0d1a')
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

for idx, row in sample.iterrows():
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.set_facecolor('#0d0d1a')

    # Parse flux and lambda
    # Parse flux and lambda
    flux_arr = np.array(row['flux'], dtype=float)
    lam_arr = np.array(row['lambda'], dtype=float)

    flux_arr = np.array(flux_arr, dtype=float)
    lam_arr  = np.array(lam_arr, dtype=float)

    label = str(row['class']).strip() if pd.notna(row['class']) else 'UNKNOWN'
    color = CLASS_COLORS.get(label, '#AAAAAA')

    # Smooth
    flux_smooth = gaussian_filter1d(flux_arr, sigma=2)

    # Plot
    ax.plot(lam_arr, flux_smooth, color=color, linewidth=0.8, alpha=0.9)
    mask = lam_arr > 3800
    ax.fill_between(lam_arr[mask], flux_smooth[mask], alpha=0.15, color=color)

    # Labels
    ax.set_xlabel('Wavelength (Å)', fontsize=7, color='#888888')
    ax.set_ylabel('Flux', fontsize=7, color='#888888')
    ax.tick_params(colors='#666666', labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    snr = f"S/N: {row['snMedian']:.1f}" if pd.notna(row['snMedian']) else ""
    z   = f"z={row['Z']:.3f}" if pd.notna(row['Z']) else ""
    ax.set_title(f"ID: {row['object_id']} — {label}  {z}  {snr}",
                 fontsize=7, color=color, pad=6, fontweight='bold')

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=c, linewidth=2, label=k)
                   for k, c in CLASS_COLORS.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           facecolor='#1a1a2e', edgecolor='#333333',
           labelcolor='white', fontsize=9, bbox_to_anchor=(0.5, 0.01))

fig.suptitle('SDSS Spectral Flux — 10 Random Objects', fontsize=14,
             color='white', fontweight='bold', y=0.98)

plt.savefig("./spectra_10.png", dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
print("Saved to ./spectra_10.png")
plt.show()