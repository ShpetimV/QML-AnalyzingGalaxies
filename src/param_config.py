# param_config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class SDSSDataConfig:
    # Paths
    parquet_path: str = 'dataset/ML_SDSS_CLEANED_DATA.parquet'
    target_col: str = 'FINAL_CLASS'

    # Feature Selection (What we actually feed the model)
    use_flux: bool = True
    scalar_cols: List[str] = field(default_factory=lambda: [
        'Z', 'SPECTROFLUX_U', 'SPECTROFLUX_G', 'SPECTROFLUX_R', 'SPECTROFLUX_I', 'SPECTROFLUX_Z'
    ])

    # Dataset Splits
    batch_size: int = 256
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    fixed_length: int = 4096
    random_state: int = 42

    # Performance
    num_workers: int = 6
    pin_memory: bool = True

    # Augmentation Parameters
    use_augmentation: bool = True
    noise_level: float = 0.005
    star_max_shift: int = 1  # Tighter for stars
    gal_max_shift: int = 5  # Larger for Galaxies/QSOs
    max_smoothing_sigma: float = 0.5
    mask_prob: float = 0.005
    scale_range: tuple = (0.9, 1.1)


@dataclass
class TrainingConfig:
    checkpoint_dir: str = './checkpoints'
    results_dir_baseline: str = './results_baseline'
    random_state: int = 42

    dropout: float = 0.3
    se_reduction: int = 16

    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0


@dataclass
class VisualConfig:
    bg_color: str = '#0d0d1a'
    text_color: str = '#cccccc'
    grid_color: str = '#333333'
    accent_color: str = '#00BFFF'
    warning_color: str = '#FFD700'
    error_color: str = '#FF6347'