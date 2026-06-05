# Quantum Machine Learning for Analyzing Astronomical Objects

Bachelor Thesis — ZHAW Zurich University of Applied Sciences, Winterthur
**Authors:** Shpetim Veseli, Moritz Feuchter

---

## Overview

This repository contains the code, models, and experimental results for the
bachelor thesis *Quantum Machine Learning for Analyzing Astronomical Objects*.
The work investigates whether **variational quantum circuits (VQCs)** and related
quantum machine-learning techniques offer any practical benefit — in accuracy,
parameter efficiency, or trainability — over classical neural networks for the
**spectral classification of astronomical objects** from the Sloan Digital Sky
Survey (SDSS).

Rather than chasing a raw-accuracy "quantum advantage," each experiment pairs a
quantum model against a **parameter-matched classical control** under an identical
training protocol, so that any observed difference is attributable to the model
class and not to confounding factors such as capacity, optimisation, or data.

## Research focus

The central questions explored quantum machine learning algorithms in the context of a real-world, high-dimensional classification task:

1. Whether a VQC can match a classical model of the same trainable-parameter
   budget on a hard binary spectral-classification task.
2. Whether a small hybrid quantum head on top of a frozen, pretrained feature
   extractor is more parameter-efficient than its classical counterparts.
3. How quantum circuit shape (qubit count vs. circuit depth) and data encoding
   affect class balance, convergence, and stability.

## Data

The dataset is derived from SDSS spectra (flux vs. wavelength), cleaned and merged
into a fixed-length representation with associated scalar features (redshift and
photometric fluxes). Classes correspond to stellar, galactic, and quasar
sub-types; the experiments focus on deliberately **hard confusion pairs and
groups** identified from the full multi-class confusion matrix. Class imbalance is
handled via a weighted sampler and a focal-loss objective. See
[`src/DataLoader_README.md`](src/DataLoader_README.md) and
[`dataset/`](dataset/) for construction and preprocessing details.

## Repository structure

```
.
├── src/                      # Library code
│   ├── models/               # Model definitions
│   │   ├── classical_cnn.py      # The "Beast": CNN+Transformer extractor/classifier
│   │   ├── quantum_model.py      # Angle-encoded VQC classifier (Experiment 1)
│   │   ├── classical_mirror.py   # Parameter-matched classical controls (Experiment 1)
│   │   └── exp3_models.py        # Frozen-extractor heads: dense / VQC / controls (Experiment 2)
│   ├── training/             # Shared trainer, metrics, logging
│   ├── sdss_dataloader.py    # Data module: splits, augmentation, sampling
│   └── param_config.py       # Central hyperparameter / data configuration
│
├── experiments/              # Runnable training & analysis scripts (see experiments/README.md)
│
├── experiment_results/       # Saved figures and metrics, organised by experiment
│   ├── baseline/             # Backbone feature extractor
│   ├── experiment1_binary/   # VQC vs. parameter-matched classical (binary)
│   └── experiment2_4class/   # Hybrid quantum head vs. classical controls (4-class)
│
├── dataset/                  # Dataset construction, download, and cleaned data
├── figures/                  # Methodology figures (e.g. circuit diagrams)
├── runs/                     # Per-run training logs and checkpoints (not version-controlled)
└── pyproject.toml            # Dependencies (managed with uv)
```

## Experiments

| # | Task | Question | Key models | Results |
|---|------|----------|-----------|---------|
| Baseline | 62-class | Reference feature extractor ("the Beast") | CNN + Transformer | `experiment_results/baseline/` |
| 1 | Binary (BROWN_DWARF_L vs. M8) | VQC vs. parameter-matched classical | Angle-encoded VQC; classical mirror (ReLU/Tanh) | `experiment_results/experiment1_binary/` |
| 2 | 4-class hard task | Parameter-efficient hybrid quantum head | Dense (A), VQC (B), classical controls (C/D) | `experiment_results/experiment2_4class/` |

Per-script documentation and run order are
in [`experiments/README.md`](experiments/README.md).

> **Methodological note.** Across experiments, an apparent quantum advantage from a
> parameter-matched classical baseline was traced to a *dead-ReLU artifact* in
> narrow (width-4) layers. Replacing the inner ReLU with a saturating activation
> (Tanh) restored parity. Reported comparisons therefore use the corrected
> controls; the artifact itself is documented as a result.

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management
(Python ≥ 3.12, < 3.14).

```bash
# Install uv (if not already installed), then sync the environment
uv sync
```

Key dependencies: PyTorch (classical models and autograd), PennyLane (quantum
circuit simulation via `default.qubit`), scikit-learn, NumPy/SciPy, and
matplotlib. The quantum circuits are simulated **noiselessly** on CPU; no physical
quantum hardware is required.

## Usage

All scripts are run from the project root so that relative paths resolve:

```bash
# 1. (Prerequisite) Train the frozen feature extractor, if its checkpoint is absent
uv run python experiments/train_baseline_cnn.py

# 2. Experiment 1 — binary VQC vs. classical (canonical multi-seed sweep)
uv run python experiments/train_sweep_experiment_binary_quantum_against_classical.py

# 3. Experiment 2 — the four 4-class heads (A: dense, B: quantum, C: ReLU, D: Tanh)
uv run python experiments/train_experiment_4class_classical.py
uv run python experiments/train_experiment_4class_quantum.py
uv run python experiments/train_experiment_4class_classical_tiny.py
uv run python experiments/train_experiment_4class_classical_tiny_tanh.py
```

Hyperparameters are set in `src/param_config.py` and in constants near the top of
each script. See [`experiments/README.md`](experiments/README.md) for the full
catalogue, recommended run order, and per-script details.

## Reproducibility

- A fixed random seed (42) controls data splitting, subsampling, and model
  initialisation; multi-seed sweeps additionally report mean ± standard deviation.
- The data split is 70 / 15 / 15 (train / validation / test); test data are used
  only for final evaluation of the best validation checkpoint.
- Exact dependency versions are pinned in `pyproject.toml` and `uv.lock`.

## Authors

- **Shpetim Veseli** — ZHAW Winterthur
- **Moritz Feuchter** — ZHAW Winterthur

## Acknowledgements

Conducted as a bachelor thesis at the ZHAW School of Engineering, Winterthur.
Spectral data courtesy of the Sloan Digital Sky Survey (SDSS).
