"""
Build a single Parquet file from the GaSNet‑II SDSS dataset for use with the
SpectraClassifier model and SDSSDataModule.

Assumes the dataset is already downloaded and its structure looks like:

    GaSNet-II-SDSS-dataset/
    ├── train.fits
    ├── valid.fits
    ├── test.fits
    └── train_data/
        ├── GALAXY_.fits
        ├── GALAXY_AGN.fits
        ...
        └── STAR_K5.fits

Usage:
    python build_gasnet_parquet.py --data_dir ./GaSNet-II-SDSS-dataset --output GasNetII_full.parquet
"""

import os
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import polars as pl
from astropy.io import fits

# ----------------------------------------------------------------------
#  Smart FITS reading – automatically detect flux and label columns
# ----------------------------------------------------------------------

# Try several common column names in order of preference
CANDIDATE_FLUX_COLS = ["int_flux", "flux", "spectrum", "spec"]
CANDIDATE_LABEL_COLS = ["CLASS", "label", "class", "type", "target"]


def _find_column(hdu, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name (case‑insensitive)."""
    colnames = [c.lower() for c in hdu.columns.names]
    for c in candidates:
        if c.lower() in colnames:
            idx = colnames.index(c.lower())
            return hdu.columns.names[idx]
    return None


def read_fits_table(filepath: str) -> Optional[List[Tuple[np.ndarray, str]]]:
    """
    Read a FITS binary table and extract flux + full subclass label.
    The full label is built as 'CLASS_SUBCLASS' (e.g. 'GALAXY_AGN').
    Returns a list of (flux_1d_array, label_string) or None on failure.
    """
    try:
        with fits.open(filepath, memmap=True) as hdul:
            table_hdu = None
            for ext in hdul:
                if isinstance(ext, fits.BinTableHDU):
                    table_hdu = ext
                    break
            if table_hdu is None:
                return None

            # Find flux column
            flux_col = _find_column(table_hdu, CANDIDATE_FLUX_COLS)
            if flux_col is None:
                return None

            # Find CLASS and SUBCLASS columns (case‑insensitive)
            colnames_lower = [c.lower() for c in table_hdu.columns.names]
            class_col = subclass_col = None
            for orig, low in zip(table_hdu.columns.names, colnames_lower):
                if low == 'class':
                    class_col = orig
                elif low == 'subclass':
                    subclass_col = orig

            if class_col is None:
                return None

            data = table_hdu.data
            records = []

            def _extract_string(val) -> str:
                """Handle FITS strings that may be numpy bytes_ arrays or regular strings."""
                if isinstance(val, np.ndarray):
                    val = val.flat[0]
                if isinstance(val, bytes):
                    return val.decode('utf-8').strip()
                return str(val).strip()

            for row in data:
                flux_arr = np.asarray(row[flux_col], dtype=np.float32).ravel()
                cls = _extract_string(row[class_col])
                subcls = _extract_string(row[subclass_col]) if subclass_col else ""
                full_label = f"{cls}_{subcls}"
                records.append((flux_arr, full_label))

            return records

    except Exception as e:
        warnings.warn(f"Could not read {filepath}: {e}")
        return None


# ----------------------------------------------------------------------
#  Main builder
# ----------------------------------------------------------------------

def build_parquet(data_dir: str, output_path: str) -> None:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    records: List[Tuple[np.ndarray, str]] = []

    # 1) Try the pre‑split combined files first
    print("Checking for combined train.fits / valid.fits / test.fits ...")
    for split_name in ("train", "valid", "test"):
        fits_path = data_dir / f"{split_name}.fits"
        if fits_path.exists():
            print(f"  Reading {fits_path} ...")
            split_records = read_fits_table(str(fits_path))
            if split_records is not None:
                print(f"    -> loaded {len(split_records)} spectra")
                records.extend(split_records)
            else:
                print(f"    -> missing expected flux/label columns; trying folder fallback for training data only")
                # Only the training split has a per‑class folder fallback
                if split_name == "train":
                    train_folder = data_dir / "train_data"
                    if train_folder.is_dir():
                        print(f"  Falling back to per‑class folder: {train_folder}")
                        folder_records = read_fits_table(str(train_folder))
                        if folder_records is None:
                            # Folder itself is not a single FITS file; read all .fits inside
                            folder_records = []
                            for class_fits in sorted(train_folder.glob("*.fits")):
                                rec = read_fits_table(str(class_fits))
                                if rec is None:
                                    raise RuntimeError(f"Failed to read {class_fits}")
                                folder_records.extend(rec)
                            print(
                                f"    -> loaded {len(folder_records)} spectra from {len(list(train_folder.glob('*.fits')))} class files")
                        records.extend(folder_records)
                    else:
                        raise RuntimeError("Neither train.fits nor train_data/ directory found.")
                else:
                    raise RuntimeError(
                        f"{fits_path} exists but does not contain the required columns."
                    )
        else:
            # combined files missing – for training, directly read the folder
            if split_name == "train":
                train_folder = data_dir / "train_data"
                if train_folder.is_dir():
                    print(f"  No train.fits found, using per‑class folder: {train_folder}")
                    folder_records = []
                    for class_fits in sorted(train_folder.glob("*.fits")):
                        rec = read_fits_table(str(class_fits))
                        if rec is None:
                            raise RuntimeError(f"Failed to read {class_fits}")
                        folder_records.extend(rec)
                    print(
                        f"    -> loaded {len(folder_records)} spectra from {len(list(train_folder.glob('*.fits')))} class files")
                    records.extend(folder_records)
                else:
                    raise RuntimeError("Neither train.fits nor train_data/ found.")
            else:
                raise FileNotFoundError(f"Required split file not found: {fits_path}")

    if not records:
        raise RuntimeError("No spectra were loaded – check dataset structure.")

    print(f"\nTotal spectra collected: {len(records)}")

    # 2) Build Polars DataFrame
    print("Building Polars DataFrame ...")
    flux_list = [f for f, _ in records]
    label_list = [lbl for _, lbl in records]

    df = pl.DataFrame({
        "FLUX": pl.Series(flux_list, dtype=pl.List(pl.Float32)),
        "FINAL_CLASS": pl.Series(label_list, dtype=pl.Utf8)
    })

    # 3) Write Parquet (use snappy for speed or zstd for smaller size)
    print(f"Writing Parquet to {output_path} ...")
    df.write_parquet(output_path, compression="zstd", statistics=True)
    print("Done.")

    # Quick sanity check
    file_size_gb = os.path.getsize(output_path) / 1e9
    print(f"Parquet file size: {file_size_gb:.2f} GB")
    test_row = pl.read_parquet(output_path, n_rows=1)
    sample_flux_len = len(test_row["FLUX"][0])
    print(f"Sample FLUX length: {sample_flux_len}")
    unique_classes = df["FINAL_CLASS"].unique().to_list()
    print(f"Classes found: {sorted(unique_classes)}")


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build GasNetII_full.parquet from the GaSNet-II-SDSS dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./GaSNet-II-SDSS-dataset",
        help="Path to the cloned/extracted dataset directory."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="GasNetII_full.parquet",
        help="Output Parquet file name."
    )
    args = parser.parse_args()

    build_parquet(args.data_dir, args.output)
