import os
import json
import numpy as np
import polars as pl
from astropy.io import fits
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SPECTRA_DIR = os.path.join(PROJECT_DIR, 'sdss_spectra')
LABELS_FILE = 'assets/my_qml_dataset_labels.parquet'
MAPPING_FILE = 'assets/subclass_merge_groups.json'
FINAL_DATASET_FILE = 'ML_SDSS_CLEANED_DATA.parquet'

MIN_SAMPLES = 250
MAX_SAMPLES = 25_000
BATCH_SIZE = 50_000
TEMP_DIR = os.path.join(PROJECT_DIR, 'temp_chunks')
os.makedirs(TEMP_DIR, exist_ok=True)


# ==========================================
# CORE FUNCTIONS
# ==========================================

def load_mapping_logic(mapping_path):
    """Creates a collision-proof reverse map for CLASS_SUBCLASS lookup."""
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    reverse_map = {}
    for group_name, subclasses in mapping.items():
        for sub in subclasses:

            if sub == "":
                sub = "NORMAL"

            # Determine prefix based on Group Name to prevent GALAXY/QSO collisions
            if group_name.startswith("GALAXY_"):
                prefix = "GALAXY"
            elif group_name.startswith("QSO_"):
                prefix = "QSO"
            else:
                prefix = "STAR"

            reverse_map[f"{prefix}_{sub}"] = group_name

    return reverse_map


def prepare_metadata(labels_path, reverse_map):
    """Loads, groups, and filters metadata before FITS extraction."""
    df = pl.read_parquet(labels_path)

    # 1. Apply Group Mapping
    df = df.with_columns(
        pl.format("{}_{}", pl.col("CLASS"), pl.col("SUBCLASS")).alias("LOOKUP_ID")
    ).with_columns(
        pl.col("LOOKUP_ID").replace_strict(reverse_map, default=pl.col("SUBCLASS")).alias("FINAL_CLASS")
    ).drop("LOOKUP_ID")

    # 2. Track stats for reporting before we filter
    pre_filter_counts = df.group_by("FINAL_CLASS").len().sort("len", descending=True)

    # 3. Filter by Minimum Samples
    valid_classes = pre_filter_counts.filter(pl.col("len") >= MIN_SAMPLES)["FINAL_CLASS"].to_list()
    df_filtered = df.filter(pl.col("FINAL_CLASS").is_in(valid_classes))

    # 4. Cap by Maximum Samples (Random shuffle first -> unbiased capping)
    df_capped = (
        df_filtered
        .sample(fraction=1.0, shuffle=True, seed=42)
        .filter(pl.int_range(0, pl.len()).over("FINAL_CLASS") < MAX_SAMPLES)
    )

    return df_capped, pre_filter_counts, valid_classes


def extract_flux_data(df, spectra_dir):
    """Iterates through FITS files and extracts 1D Flux arrays."""
    current_batch_flux = []
    current_batch_indices = []
    chunk_count = 0
    rows = df.to_dicts()
    for idx, row in tqdm(enumerate(rows), total=len(rows), desc="Extracting Spectra"):
        spec_file = row['SPEC_FILE']

        try:
            parts = spec_file.split('-')
            field_folder = parts[1]
            mjd_folder = parts[2]
            file_path = os.path.join(spectra_dir, field_folder, mjd_folder, spec_file)
        except IndexError:
            continue

        if not os.path.exists(file_path):
            continue

        try:
            with fits.open(file_path) as hdul:
                # HDU 1 contains the 'flux' column in SDSS BOSS/eBOSS data
                flux = hdul[1].data['flux'].astype(np.float32).tolist()
                current_batch_flux.append(flux)
                current_batch_indices.append(idx)
        except Exception as e:
            print(f"\n[ERROR] {row['SPEC_FILE']}: {e}")
            continue

        # save chunk
        if len(current_batch_flux) >= BATCH_SIZE:
            save_chunk(df, current_batch_indices, current_batch_flux, chunk_count)
            # free memory
            current_batch_flux = []
            current_batch_indices = []
            chunk_count += 1

    #save final chunk
    if current_batch_flux:
        save_chunk(df, current_batch_indices, current_batch_flux, chunk_count)

    # final merge
    print("\nMerging chunks...")
    final_df = pl.read_parquet(f"{TEMP_DIR}/chunk_*.parquet")

    # clean up temp files
    for file in os.listdir(TEMP_DIR):
        os.remove(os.path.join(TEMP_DIR, file))

    return final_df

def save_chunk(original_df, indices, flux_data, chunk_id):
    """Helper to write a portion of data to disk."""
    chunk_df = original_df[indices]
    chunk_df = chunk_df.with_columns(pl.Series("FLUX", flux_data))
    chunk_df.write_parquet(f"{TEMP_DIR}/chunk_{chunk_id}.parquet")


def print_data_analysis(pre_stats, valid_classes, final_df):
    """Prints a detailed summary of the dataset state."""
    print("\n" + "=" * 50)
    print("-----DATASET ANALYSIS REPORT-----")
    print("=" * 50)

    # 1. Dropped Classes
    dropped = pre_stats.filter(~pl.col("FINAL_CLASS").is_in(valid_classes))
    print(f"\nCLASSES DROPPED (Below {MIN_SAMPLES} samples):")
    if dropped.height == 0:
        print("None.")
    for row in dropped.iter_rows(named=True):
        print(f"   - {row['FINAL_CLASS']}: {row['len']} samples")

    # 2. Kept Classes
    print(f"\nCLASSES KEPT ({len(valid_classes)} total):")
    final_counts = final_df.group_by("FINAL_CLASS").len().sort("len", descending=True)
    for row in final_counts.iter_rows(named=True):
        print(f"   - {row['FINAL_CLASS']:<25} | {row['len']:>5} samples")

    print("\n" + "=" * 50)
    print(f"TOTAL FINAL SAMPLES: {final_df.height}")
    print("=" * 50 + "\n")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Load Mapping
    rev_map = load_mapping_logic(MAPPING_FILE)

    # Prepare Metadata
    metadata_df, pre_stats, kept_classes = prepare_metadata(LABELS_FILE, rev_map)

    # Extract Physics Data
    dataset_df = extract_flux_data(metadata_df, SPECTRA_DIR)

    # Save
    dataset_df.write_parquet(FINAL_DATASET_FILE)

    # Report Results
    print_data_analysis(pre_stats, kept_classes, dataset_df)
    print(f"Successfully saved: {FINAL_DATASET_FILE}")