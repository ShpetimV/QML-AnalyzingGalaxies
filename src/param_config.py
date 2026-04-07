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
    batch_size: int = 64
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42

    # Performance
    num_workers: int = 4

    # Augmentation Parameters
    use_augmentation: bool = True
    noise_level: float = 0.02
    star_max_shift: int = 2  # Tighter for stars
    gal_max_shift: int = 10  # Larger for Galaxies/QSOs
    max_smoothing_sigma: float = 1.0
    mask_prob: float = 0.02
    scale_range: tuple = (0.9, 1.1)