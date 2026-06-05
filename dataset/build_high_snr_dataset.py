import polars as pl
import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from pathlib import Path
from typing import cast

# --- CONFIGURATION ---
INPUT_PARQUET = "subset_GasNetII_replica.parquet"  # Output from build_subset_parquet.py
OUTPUT_PARQUET = "GasNetII_replica.parquet"
SPECTRA_DIR = "sdss_spectra"
TARGET_CLASSES = ["STAR_A0", "STAR_F5", "STAR_F9", "STAR_G2", "STAR_K1", "STAR_K3", "STAR_K5", "GALAXY_NORMAL",
                  "GALAXY_AGN", "GALAXY_STARBURST", "GALAXY_STARFORMING", "QSO_NORMAL", "QSO_BROADLINE"]
MIN_SNR = 0
MAX_SAMPLES_PER_CLASS = 20000


def extract_flux_from_fits(fits_path):
    """Opens a FITS file and returns the flux array."""
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            flux = hdul[1].data['flux'].astype(np.float32)
            return flux.tolist()
    except Exception:
        return None


def main():
    print(f"Loading {INPUT_PARQUET}...")

    # 1. Lazy load the dataset
    df = pl.scan_parquet(INPUT_PARQUET)

    # 2. Filter by our isolated classes and high SNR
    print(f"Filtering for classes: {TARGET_CLASSES} with snMedian >= {MIN_SNR}...")
    filtered_df = df.filter(
        (pl.col("FINAL_CLASS").is_in(TARGET_CLASSES)) &
        (pl.col("SN_MEDIAN_ALL") >= MIN_SNR)
    ).collect()

    if len(filtered_df) == 0:
        print("ERROR: No samples found! Check your column names or lower the MIN_SNR.")
        return

    # 3. Check the counts
    counts = filtered_df["FINAL_CLASS"].value_counts().sort("count", descending=True)
    print("\nAvailable High-SNR Samples per class:")
    print(counts)

    # 4. Perfectly balance and cap the dataset
    print(f"\nBalancing and capping at {MAX_SAMPLES_PER_CLASS} samples per class...")
    min_available_raw = counts["count"].min()
    if min_available_raw is None:
        print("ERROR: No samples available after filtering.")
        return

    min_available = cast(int, min_available_raw)
    samples_to_take = int(min(min_available, int(MAX_SAMPLES_PER_CLASS)))

    print(f"Taking exactly {samples_to_take} highest-SNR samples from each class to ensure perfect balance.")

    final_dfs = []
    for cls in TARGET_CLASSES:
        cls_df = (
            filtered_df
            .filter(pl.col("FINAL_CLASS") == cls)
            .drop_nulls(["SN_MEDIAN_ALL"])
            .sort(["SN_MEDIAN_ALL", "SPEC_FILE"], descending=[True, False])
            .head(int(samples_to_take))
        )

        if len(cls_df) < samples_to_take:
            print(
                f"Warning: only {len(cls_df)} usable samples found for {cls} after ranking and missing-value cleanup.")

        final_dfs.append(cls_df)

    final_balanced_df = pl.concat(final_dfs)

    print(f"\nIndexing all FITS files in {SPECTRA_DIR} and its subfolders... (This takes a few seconds)")
    fits_map = {}
    for p in Path(SPECTRA_DIR).rglob("*.fits"):
        fits_map[p.name] = str(p)
    print(f"Found {len(fits_map)} total FITS files locally.")

    # 5. Extract FLUX from the FITS files
    print(f"\nExtracting FLUX arrays from {len(final_balanced_df)} targeted files...")

    flux_data = []
    valid_mask = []
    missing_count = 0

    rows = final_balanced_df.to_dicts()

    for row in tqdm(rows, desc="Processing FITS"):
        base_name = os.path.basename(row["SPEC_FILE"])

        fits_path = fits_map.get(base_name)

        if fits_path is not None:
            flux = extract_flux_from_fits(fits_path)
            if flux is not None:
                flux_data.append(flux)
                valid_mask.append(True)
                continue

        flux_data.append(None)
        valid_mask.append(False)
        missing_count += 1

    if missing_count > 0:
        print(f"\nWarning: Could not find or read {missing_count} FITS files. Dropping those rows.")

    # 6. Add the FLUX column and drop any failed reads
    final_balanced_df = final_balanced_df.with_columns(
        pl.Series("FLUX", flux_data)
    )
    final_balanced_df = final_balanced_df.filter(pl.Series(valid_mask))

    # 7. Save the new dataset
    print(f"\nSaving final dataset ({final_balanced_df.height} total rows) to {OUTPUT_PARQUET}...")
    final_balanced_df.write_parquet(OUTPUT_PARQUET)
    print("Done! Dataset is fully built and ready for DataLoader.")


if __name__ == "__main__":
    main()
