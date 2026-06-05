# Experiments

This is a quick guide to the training scripts in `experiments/`. The scripts are mostly standalone, so you can pick and choose which ones to run based on which results you want to reproduce.

## How to run

Run everything **from the project root** (not from inside `experiments/`), so the
relative paths (`dataset/`, `src/models/trained_models/`, `results_*`) resolve:

```bash
cd /Users/shpetim/git/FS26/QML-AnalyzingGalaxies
python experiments/<script>.py
```

- Most scripts read hyperparameters from constants near the top of the file
  (`N_QUBITS`, `N_LAYERS`, `CLASSES`, ...) and from `src/param_config.py`
  (`TrainingConfig.epochs`, lr, etc.). Edit those, then run.
- A few scripts take CLI flags — only `train_experiment_4class_classical_tiny_tanh.py`
  currently supports `--epochs` and `--out-dir`. The rest use the in-file constants.
- The two `*sweep*` scripts have an `EDIT THIS BLOCK` section: set `MODEL` and the
  config/seed lists, then run once per `MODEL` value.
- Quantum scripts run the circuit on PennyLane's `default.qubit` (CPU, noiseless),
  so they are much slower than the classical ones (seconds/epoch).

## Prerequisite — train the backbone first

Every Experiment-3 model uses a **frozen** pretrained extractor ("the Beast") at
`src/models/trained_models/baseline_cnn_transformer.pt`. It already exists in the
repo. If it is ever missing, regenerate it with:

```bash
python experiments/train_baseline_cnn.py     # writes runs/Baseline_CNN_*/
```

then copy/rename the best checkpoint to the path above.

---

## Experiment 1 — Binary: VQC vs. parameter-matched classical

Task: `STAR_BROWN_DWARF_L` vs `STAR_M8`. **Canonical numbers come from the sweep**
(`results_sweep_binary/SWEEP_REPORT.md`), not the single-run scripts.

| Script | Role | Output |
|---|---|---|
| `train_baseline_binary.py` | Full CNN, from scratch — high-capacity classical reference | `results_classical_binary/` |
| `train_quantum_binary.py` | Single VQC run (set `N_QUBITS`/`N_LAYERS`) | `results_quantum_binary_*/` |
| `train_classical_mirror_binary.py` | Param-matched classical, **ReLU** — shows the dead-ReLU collapse (~46%) | `results_classical_mirror_binary_*/` |
| `train_classical_mirror_binary_tanh.py` | Param-matched classical, **Tanh** — the dead-ReLU fix; restores parity | `results_classical_mirror_tanh_binary_*/` |
| **`train_sweep_experiment_binary_quantum_against_classical.py`** | **Canonical** multi-config, multi-seed sweep (set `MODEL`) | `results_sweep_binary/` |

To reproduce Experiment 1: run the sweep with `MODEL="quantum"`, then again with
`MODEL="classical"`. The single-run scripts above are for inspecting one config /
producing per-model plots.

## Experiment 3 — 4-class: the parameter-efficiency showdown

Task: `STAR_BROWN_DWARF_L`, `STAR_M8`, `GALAXY_STARBURST`, `GALAXY_STARFORMING`,
1000 samples/class, on top of the frozen Beast. **Four headline models (A–D):**

| Model | Script | Head | Params | Output |
|---|---|---|---|---|
| **A** | `train_experiment_4class_classical.py` | Dense `Linear(128,38)→ReLU→Linear(38,4)` | ~5058 | `results_exp3_classical_4class_100epochs/` |
| **B** | `train_experiment_4class_quantum.py` | `Linear(128,4)→VQC(4q,5L)→Linear(4,4)` | 556 | `results_exp3_quantum_4class_100epochs/` |
| **C** | `train_experiment_4class_classical_tiny.py` | Classical control, **ReLU** inner | 556 | `results_exp3b_tiny_classical_4class/` |
| **D** | `train_experiment_4class_classical_tiny_tanh.py` | Classical control, **Tanh** inner | 556 | `results_exp3b_tiny_classical_tanh_4class/` |

**Key result:** B ≈ D ≈ A ≈ 96%; C caps at ~79% — but C's failure is a dead-ReLU
artifact (D, identical except ReLU→Tanh, recovers fully). So Experiment 3 is
**quantum–classical parity**, not a quantum advantage. See `../EXP3_ANALYSIS.md`.

To reproduce, run A, B, C, D (D supports `--epochs 100`). For matched 100-epoch
runs of A/B/C, set `TrainingConfig.epochs = 100` in `src/param_config.py` first.

### Experiment 3 — ablations (supporting, not headline)

| Script | What it explores | Output |
|---|---|---|
| `train_experiment_4class_quantum_pca_hybrid.py` | PCA(128→64)+Linear bottleneck; shows PCA-fed VQC plateaus ~76% | `results_exp3_quantum_hybrid_*/` |
| `train_experiment_4class_quantum_tiny.py` | Richer ansatz (`FrozenBeastVQCClassifier2`: RX/RZ + Rot + alt. CNOT) | `results_exp3b_quantum_4class/` |
| `train_sweep_experiment_4class_pca_version.py` | Frozen-PCA sweep, VQC vs tiny MLP across widths (set `MODEL`) | `results_sweep_exp3_pca/` |

## Other

| Script | Role |
|---|---|
| `train_baseline_cnn.py` | **Prerequisite** — trains the frozen backbone (see above) |
| `train_grad_cam_quantum.py` | **Analysis, not training** — Grad-CAM / saliency heatmaps for a trained quantum binary model |

---


