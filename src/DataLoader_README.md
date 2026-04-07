## 1. Quick Start
Initialize the DataModule in your main training script. All parameters are managed in `src/param_config.py`.

```python
from src.param_config import SDSSDataConfig
from src.sdss_dataloader import SDSSDataModule

# Initialize
config = SDSSDataConfig(batch_size=64)
dm = SDSSDataModule(config)
dm.prepare_data()

# Loaders
train_loader = dm.get_loader(dm.train_ds, use_sampler=True) # Balanced classes
val_loader = dm.get_loader(dm.val_ds)
```

---

## 2. Batch Data Format
The `DataLoader` returns a dictionary for every batch:
* **`batch['flux']`**: Tensor of shape **$[Batch, 1, 4448]$**. (Cropped & Scaled).
* **`batch['label']`**: Tensor of shape **$[Batch]$**. (Long).
* **`batch['scalars']`**: Tensor of shape **$[Batch, 6]$**. (Z, Flux_U, G, R, I, Z).

---

## 3. Model Implementation Snippets

### A. Classical 1D CNN
Standard usage. Flux is already formatted for `Conv1d` layers.
```python
for batch in train_loader:
    x = batch['flux'].to(device)  # [64, 1, 4448]
    y = batch['label'].to(device)
    logits = model(x)
```

### B. QML: Angle Encoding
Requires downsampling. Use `AdaptiveAvgPool1d` to match your qubit count (e.g., 8). or oher smart feature selection method to reduce dimensionality while preserving information.
```python
import torch.nn.functional as F

for batch in train_loader:
    x = batch['flux'].to(device)
    # Squash 4448 -> 8 features
    x_reduced = F.adaptive_avg_pool1d(x, output_size=8).squeeze(1) 
    # Use x_reduced values as rotation angles in qubits
    logits = quantum_model(x_reduced)
```

### C. QML: Amplitude Encoding
Requires padding to the nearest power of 2 ($2^n$) and L2 normalization.
```python
for batch in train_loader:
    x = batch['flux'].to(device) # [64, 1, 4448]
    # Pad to 8192 (for 13 qubits)
    x_padded = F.pad(x, (0, 8192 - x.shape[-1]))
    # Normalize (Unit Vector)
    x_norm = F.normalize(x_padded, p=2, dim=-1).squeeze(1)
    logits = quantum_model(x_norm)
```

### D. Quanvolutional Neural Network (Quanv)
Similar to CNN. The Quanv layer slides a quantum circuit across the signal.
```python
for batch in train_loader:
    x = batch['flux'].to(device) # [64, 1, 4448]
    # Quanv layer expects [Batch, 1, Length]
    features = quanv_layer(x)
    logits = final_classifier(features)
```

---

## 4. Key Performance Notes
* **Class Imbalance:** Handled automatically by `WeightedRandomSampler` in the `get_loader` call.
* **Augmentation:** Only active on `train_ds`. Includes Redshift-shift, Gaussian noise, Masking, and Scaling.
* **Device Transfer:** Use `pin_memory=True` in the loader if using an NVIDIA GPU. For Mac (MPS), keep it `False`.
* **Num Workers:** Set `num_workers=4` in config to prevent CPU bottlenecks during augmentation.

---

## 5. Troubleshooting Imports
Always run from the **project root** using:
`uv run python -m src.your_script_name`