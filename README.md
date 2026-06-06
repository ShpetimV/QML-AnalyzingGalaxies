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
4. How QSVMs can be trained in a few-shot area and how they compare to classical
   SVMs.
5. Whether quanvolutional layers can be robust to adversarial and ood training with added noise in testing.
6. How a quantum CNN can be trained on a hard 4-class spectral-classification task.

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
├── src/                          # Library code
│   ├── models/                   # Model definitions
│   │   ├── classical_cnn.py      # Baseline CNN+Transformer backbone & classifier
│   │   ├── quantum_model.py      # Angle/amplitude encoded VQC (Experiment 1)
│   │   ├── classical_mirror.py   # Parameter-matched classical nets (Experiment 1)
│   │   ├── exp3_models.py        # Frozen-extractor heads: dense, VQC, controls (Exp 2)
│   │   ├── qcnn.py               # Quantum convolutional neural network (Experiment 5)
│   │   ├── quanvolution.py       # Quanvolutional layer & noise‑robust models (Exp 4)
│   │   ├── qsvm.py               # Quantum kernel SVM (Experiment 3 few‑shot)
│   │   └── classical_baselines.py# Tiny classical heads for Exp2/Exp5 controls
│   ├── training/                 # Shared training utilities
│   │   ├── trainer.py            # Main training loop (classification & robustness)
│   │   ├── metrics.py            # Accuracy, confusion matrix, AUC, Jacobian slopes
│   │   └── logger.py             # TensorBoard / file logging, checkpointing
│   ├── sdss_dataloader.py        # Data module: SDSS, GasNet, splits, PCA, augmentations
│   └── param_config.py           # Central config: hyperparams, paths, experiment flags
│
├── experiments/                  # Runnable training & analysis scripts (see experiments/README.md)
│
├── experiment_results/           # Saved figures and metrics, organised by experiment
│   ├── Baseline_CNN/             # Backbone feature extractor
│   ├── GasNet_II_Replica/        # Replicated baseline on GasNet II dataset
│   ├── experiment1_binary/       # VQC vs. classical mirror (binary L vs M8)
│   ├── experiment2_4class/       # Hybrid quantum head vs. classical controls (4‑class)
│   ├── experiment3_fewshot/      # QSVM few‑shot scaling behaviour
│   ├── experiment4_quanv_noise_robustness/ # Quanvolution under adversarial/OOD noise
│   └── experiment5_qcnn/         # QCNN vs. classical controls (CNN_Huge, CNN_Tiny)
│
├── dataset/                      # Dataset construction, download, and cleaned data
│
├── pyproject.toml                # Dependencies (managed with uv)
├── uv.lock                       # Exact dependency lockfile
├── setup_data_pipeline.sh        # Script to initialise data & environment
├── run_exp4_parallel_noise.sh    # Launch parallel Exp4 noise jobs
└── run_exp5_qcnn_matrix.sh       # Launch full Exp5 matrix experiment
```

## Experiments

| # | Task | Question | Key models                                      | Results |
|---|------|----------|-------------------------------------------------|---------|
| Baseline | 62-class | Reference feature extractor ("the Beast") | CNN + Transformer                               | `experiment_results/baseline/` |
| 1 | Binary (BROWN_DWARF_L vs. M8) | VQC vs. parameter-matched classical | Angle-encoded VQC; classical mirror (ReLU/Tanh) | `experiment_results/experiment1_binary/` |
| 2 | 4-class hard task | Parameter-efficient hybrid quantum head | Dense (A), VQC (B), classical controls (C/D)    | `experiment_results/experiment2_4class/` |
| 3 | Few-shot scaling | Quantum SVM | SVM (A), QSVM (B)                               | `experiment_results/experiment3_fewshot/` |
| 4 | Noise robustness | Quanvolutional layer | classic CNN (A), Quanv (B)                      | `experiment_results/experiment4_quanv_noise_robustness/` |
| 5 | 4-class hard task | Quantum CNN | classical CNN (A), QCNN (B), tiny classical control (C) | `experiment_results/experiment5_qcnn/` |

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

The dataset afterwards needs to be downloaded and processed:

```bash
bash setup_data_pipeline.sh
```

This script will download the SDSS spectra, clean and merge them into a a full dataset.
For the other experiments, subsets of the full dataset have to be created.

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

# 4. Experiment 3 — few-shot scaling of SVM vs. QSVM
# n-feature sweep: 
for f in 4 6 8 12 16; do uv run python -m experiments.run_exp1_fewshot_new --features $f --out experiment_results/experiment3_fewshot/results_sweep/f$f; done
# c-value sweep:
for c in 0.1 0.25 0.5 0.75 1.0 1.5; do uv run python -m experiments.run_exp1_fewshot_new --features 12 --bandwidth $c --out experiment_results/experiment3_fewshot/results_bw/c$c; done

# 5. Experiment 4 — noise robustness of quanvolutional layer
uv run bash experiments/run_exp4_parallel_noise.sh

# 6. Experiment 5 — QCNN vs. classical controls (full matrix)
uv run bash experiments/run_exp5_qcnn_matrix.sh
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
