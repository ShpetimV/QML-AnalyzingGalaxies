import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.param_config import SDSSDataConfig
import pyarrow.parquet as pq
import gc


class SDSSDataset(Dataset):
    def __init__(self, full_flux, full_scalars, full_labels, indices, class_names, config: SDSSDataConfig,
                 is_train: bool = False):
        # reference to dataset instead of copying into RAM
        self.full_flux = full_flux
        self.full_scalars = full_scalars
        self.full_labels = full_labels
        self.indices = indices
        self.class_names = class_names
        self.config = config
        self.is_train = is_train

        # edge artifacts
        self.start_trim = 100
        self.end_trim = 100

    def __len__(self):
        return len(self.indices)

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
        real_idx = self.indices[idx]
        flux_row = self.full_flux[real_idx]
        cropped_flux = flux_row[self.start_trim: -self.end_trim]

        flux = torch.tensor(cropped_flux, dtype=torch.float32)
        label = torch.tensor(self.full_labels[real_idx], dtype=torch.long)

        # normalize
        flux = (flux - flux.mean()) / (flux.std() + 1e-6)

        if self.is_train and self.config.use_augmentation:
            flux = self.apply_augmentation(flux, label.item())

        target_len = self.config.fixed_length
        current_len = flux.shape[0]

        if current_len > target_len:
            flux = flux[:target_len]
        elif current_len < target_len:
            pad_size = target_len - current_len
            flux = torch.nn.functional.pad(flux, (0, pad_size), value=0.0)

        # no scalars for now -> add them maybe later for multimodal experiments
        return {
            'flux': flux.unsqueeze(0),
            'scalars': torch.tensor([]),
            'label': label
        }


class SDSSDataModule:
    def __init__(self, config: SDSSDataConfig):
        self.config = config
        self.label_encoder = LabelEncoder()

    def prepare_data(self):
        print("Initializing memory-safe data loading...")

        parquet_file = pq.ParquetFile(self.config.parquet_path)
        num_samples = parquet_file.metadata.num_rows

        first_row = pl.read_parquet(self.config.parquet_path, n_rows=1)
        seq_length = len(first_row['FLUX'][0])

        print(f"Pre-allocating {num_samples} samples x {seq_length} length (~10.5GB)...")
        flux_data = np.empty((num_samples, seq_length), dtype=np.float32)

        print("Streaming batches into NumPy (avoids RAM spike)...")
        current_idx = 0
        # process in chunks of 50,000 rows to keep RAM usage low
        for batch in parquet_file.iter_batches(batch_size=50000, columns=['FLUX']):
            # convert batch to Polars for easy extraction of the list column
            temp_df = pl.from_arrow(batch)

            flux_slice = np.vstack(temp_df['FLUX'].to_numpy())
            batch_len = len(flux_slice)
            flux_data[current_idx: current_idx + batch_len] = flux_slice

            current_idx += batch_len
            print(f" Loaded {current_idx}/{num_samples} samples...")

            # cleanup RAM
            del temp_df
            del flux_slice
            gc.collect()

        # load labels and scalars separately (minimal RAM usage)
        print("Loading labels and scalars...")
        full_df = pl.read_parquet(self.config.parquet_path, columns=[self.config.target_col] + self.config.scalar_cols)

        labels = self.label_encoder.fit_transform(full_df[self.config.target_col].to_numpy())
        self.classes = self.label_encoder.classes_
        self.num_classes = len(self.classes)
        scalar_data = full_df.select(self.config.scalar_cols).to_numpy().astype(np.float32)

        # cleanup RAM
        del full_df
        gc.collect()
        print("Splitting datasets...")

        indices = np.arange(len(labels))
        train_val_idx, test_idx = train_test_split(indices, test_size=self.config.test_size, stratify=labels,
                                                   random_state=self.config.random_state)

        val_ratio = self.config.val_size / (self.config.train_size + self.config.val_size)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, stratify=labels[train_val_idx],
                                              random_state=self.config.random_state)

        print("Initializing Zero-Copy Datasets...")
        self.train_ds = SDSSDataset(flux_data, scalar_data, labels, train_idx, self.classes, self.config, is_train=True)
        self.val_ds = SDSSDataset(flux_data, scalar_data, labels, val_idx, self.classes, self.config, is_train=False)
        self.test_ds = SDSSDataset(flux_data, scalar_data, labels, test_idx, self.classes, self.config, is_train=False)

        print(f"Data prepared: {len(self.train_ds)} train, {len(self.val_ds)} val, {len(self.test_ds)} test samples.")

    def get_loader(self, dataset, use_sampler=False):
        sampler = None
        if use_sampler:
            # Weighted Sampler logic to balance 250 vs 25_000 samples
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