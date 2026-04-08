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
    batch_size: int = 512
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    fixed_length: int = 4096
    random_state: int = 42

    # Performance
    num_workers: int = 8
    pin_memory: bool = False

    # Augmentation Parameters
    use_augmentation: bool = True
    noise_level: float = 0.02
    star_max_shift: int = 2  # Tighter for stars
    gal_max_shift: int = 10  # Larger for Galaxies/QSOs
    max_smoothing_sigma: float = 1.0
    mask_prob: float = 0.02
    scale_range: tuple = (0.9, 1.1)


@dataclass
class TrainingConfig:
    checkpoint_dir: str = './checkpoints'
    results_dir_baseline: str = './results_baseline'
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10
    random_state: int = 42

    dropout: float = 0.3
    se_reduction: int = 16

    batch_size: int = 128
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_workers: int = 4


@dataclass
class VisualConfig:
    bg_color: str = '#0d0d1a'
    text_color: str = '#cccccc'
    grid_color: str = '#333333'
    accent_color: str = '#00BFFF'
    warning_color: str = '#FFD700'
    error_color: str = '#FF6347'