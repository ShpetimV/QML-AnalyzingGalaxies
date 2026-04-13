import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.param_config import SDSSDataConfig
import math


class SDSSDataset(Dataset):
    def __init__(self, flux_data, scalar_data, labels, class_names, config: SDSSDataConfig, is_train: bool = False):
        self.flux_data = torch.tensor(flux_data, dtype=torch.float32) if flux_data is not None else None

        # remove edge artifacts
        start_trim = 100
        end_trim = 100

        if flux_data is not None:
            cropped_flux = flux_data[:, start_trim:-end_trim]
            self.flux_data = torch.tensor(cropped_flux, dtype=torch.float32)
        else:
            self.flux_data = None

        self.scalar_data = torch.tensor(scalar_data, dtype=torch.float32) if scalar_data is not None else None
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.class_names = class_names  # Used to check for 'STAR' vs 'GALAXY'
        self.config = config
        self.is_train = is_train

    def __len__(self):
        return len(self.labels)

    def apply_augmentation(self, x, label_idx):
        """Scientific Augmentation with Class-Aware Shifting"""
        class_name = self.class_names[label_idx]

        # 1. Redshift Shift (Class-Aware)
        # We simulate (1+z) by rolling pixels.
        max_s = self.config.star_max_shift if "STAR" in class_name else self.config.gal_max_shift
        # generator = np.random.default_rng(42)
        shift = int(torch.randn(1).item() * max_s)
        shift = max(-max_s, min(max_s, shift))  # Clip to max shift range
        x = torch.roll(x, shifts=shift, dims=0)

        # 2. Gaussian Noise
        noise = torch.randn_like(x) * self.config.noise_level * x.std()
        x = x + noise

        # 3. Flux Scaling
        low, high = self.config.scale_range
        scale = low + torch.rand(1).item() * (high - low)
        x = x * scale

        # 4. Fiber Masking
        mask = torch.rand_like(x) > self.config.mask_prob
        x = x * mask.float()

        # 5. Smoothing (Gaussian blur)
        sigma = torch.rand(1).item() * self.config.max_smoothing_sigma
        if sigma > 0.1:
            kernel_size = int(sigma * 4) + 1
            if kernel_size % 2 == 0: kernel_size += 1
            kernel = torch.tensor(
                np.exp(-np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1) ** 2 / (2 * sigma ** 2)),
                dtype=torch.float32
            )
            kernel /= kernel.sum()
            x = torch.nn.functional.conv1d(x.view(1, 1, -1), kernel.view(1, 1, -1), padding=kernel_size // 2).view(-1)

        return x

    def __getitem__(self, idx):
        label = self.labels[idx]
        flux = self.flux_data[idx].clone() # avoid in-place changes to original data

        # Normalize per-sample
        flux = (flux - flux.mean()) / (flux.std() + 1e-6)

        # maybe for better normalization but probably not needded
        # median = flux.median()
        # q75, q25 = torch.quantile(flux, 0.75), torch.quantile(flux, 0.25)
        # iqr = q75 - q25
        # flux = (flux - median) / (iqr + 1e-6)
        # flux = torch.clamp(flux, -10, 10)  # Clip extreme outliers

        if self.is_train and self.config.use_augmentation:
            flux = self.apply_augmentation(flux, label.item())

        target_len = self.config.fixed_length
        current_len = flux.shape[0]

        # Handle fixed length by cropping or padding as needed
        if current_len > target_len:
            flux = flux[:target_len]  # Crop right
        elif current_len < target_len:
            pad_size = target_len - current_len
            flux = torch.nn.functional.pad(flux, (0, pad_size), value=0.0)  # Pad right with 0

        # Return a dictionary. Your model loop picks what it needs.
        return {
            'flux': flux.unsqueeze(0),  # CNNs love [Channels, Length]
            'scalars': torch.tensor([]), # Placeholder for scalar features if needed
            'label': label
        }


class SDSSDataModule:
    def __init__(self, config: SDSSDataConfig):
        self.config = config
        self.label_encoder = LabelEncoder()

    def prepare_data(self):
        df = pl.read_parquet(self.config.parquet_path)

        # Encoding and storing class names for the augmentation logic
        labels = self.label_encoder.fit_transform(df[self.config.target_col].to_numpy())
        self.classes = self.label_encoder.classes_
        self.num_classes = len(self.classes)

        flux_data = np.vstack(df['FLUX'].to_numpy())
        scalar_data = df.select(self.config.scalar_cols).to_numpy()

        indices = np.arange(len(labels))
        train_val_idx, test_idx = train_test_split(indices, test_size=self.config.test_size, stratify=labels,
                                                   random_state=self.config.random_state)

        val_ratio = self.config.val_size / (self.config.train_size + self.config.val_size)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, stratify=labels[train_val_idx],
                                              random_state=self.config.random_state)

        # Build datasets with the class list passed in
        self.train_ds = SDSSDataset(flux_data[train_idx], scalar_data[train_idx], labels[train_idx], self.classes,
                                    self.config, is_train=True)
        self.val_ds = SDSSDataset(flux_data[val_idx], scalar_data[val_idx], labels[val_idx], self.classes, self.config,
                                  is_train=False)
        self.test_ds = SDSSDataset(flux_data[test_idx], scalar_data[test_idx], labels[test_idx], self.classes,
                                   self.config, is_train=False)

    def get_loader(self, dataset, use_sampler=False):
        sampler = None
        if use_sampler:
            # Weighted Sampler logic to balance 250 vs 5000 samples
            labels = dataset.labels.numpy()
            class_sample_count = np.array([len(np.nonzero(labels == t)[0]) for t in np.unique(labels)])
            weight = 1. / class_sample_count
            samples_weight = torch.from_numpy(np.array([weight[t] for t in labels]))
            sampler = WeightedRandomSampler(samples_weight.double(), len(samples_weight))

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None and dataset.is_train),  # Shuffle if no sampler
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )