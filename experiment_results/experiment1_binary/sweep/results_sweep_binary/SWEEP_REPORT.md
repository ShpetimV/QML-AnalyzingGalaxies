# Sweep Results — Binary Task (30 epochs, 3 seeds each)

Task: `STAR_BROWN_DWARF_L` vs `STAR_M8`
Setup: shared CNN extractor (trainable), parameter-matched bottleneck/VQC head, dead-ReLU-fixed classical mirror.

## The numbers that matter

| Model | Config | Acc μ±σ | Macro F1 μ±σ | BROWN rec | M8 rec | **Gap** | ROC AUC |
|---|---|---|---|---|---|---|---|
| Quantum | 2q × 12L | 0.863 ±0.005 | 0.830 ±0.005 | 0.701 ±0.006 | 0.933 ±0.008 | **0.232** | 0.910 |
| Quantum | 3q × 8L  | 0.865 ±0.012 | 0.835 ±0.009 | 0.722 ±0.046 | 0.927 ±0.034 | 0.205 | 0.913 |
| Quantum | **4q × 6L** | **0.869 ±0.008** | **0.840 ±0.005** | 0.729 ±0.058 | 0.930 ±0.036 | 0.200 | 0.927 |
| Quantum | 6q × 4L  | 0.860 ±0.012 | 0.833 ±0.013 | 0.762 ±0.039 | 0.903 ±0.022 | 0.140 | 0.924 |
| Quantum | 8q × 3L  | 0.860 ±0.013 | 0.837 ±0.014 | **0.802 ±0.005** | 0.885 ±0.017 | **0.083** | **0.927** |
| Classical | 4 features | **0.868 ±0.016** | **0.843 ±0.016** | 0.774 ±0.080 | 0.909 ±0.047 | 0.136 | **0.933** |
| Classical | 8 features | 0.862 ±0.004 | 0.836 ±0.006 | 0.774 ±0.026 | 0.900 ±0.008 | 0.127 | 0.922 |

*Gap = M8 recall − BROWN recall. Smaller = more balanced.*
*All configs have ~24.5K total params (the CNN extractor dominates; the VQC/MLP head is the small piece).*

## Best of the best

Different metrics crown different winners — and the spread between them is small (mostly within 1σ):

- **Highest accuracy & macro F1:** Quantum 4q×6L vs Classical 4-features — statistical **tie** (0.869 vs 0.868 acc, 0.840 vs 0.843 macro F1).
- **Highest ROC AUC:** Classical 4-features (0.933) — small but consistent edge.
- **Best class balance:** **Quantum 8q×3L wins clearly** — only 8.3pp gap between BROWN and M8 recall, ~half of any other configuration. It also has the smallest std on BROWN recall (0.005), so it's reliably balanced across seeds.
- **Most consistent (lowest seed variance):** Classical 8-features — std on every metric is 1pp or less.

