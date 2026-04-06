import os
import re
import json
import numpy as np
import polars as pl
from astropy.table import Table
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FITS = 'assets/spAll-lite-v6_1_3.fits.gz'
OVERRIDE_FILE = 'assets/custom_star_name_overrides.json'
OUTPUT_MAPPING = 'assets/subclass_merge_groups.json'


def load_overrides(path):
    """Loads custom naming map from JSON if it exists."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def read_sdss_fits(path):
    """Reads SDSS FITS, handles byte-order, and converts to Polars."""
    print(f"Reading {path}...")
    astropy_table = Table.read(path, hdu=1)

    data_dict = {}
    for col in astropy_table.colnames:
        arr = np.array(astropy_table[col])
        # FITS is big-endian (>); Polars needs native (=) byte order
        if hasattr(arr, 'dtype') and arr.dtype.byteorder == '>':
            arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
        data_dict[col.upper()] = arr

    df = pl.DataFrame(data_dict)

    # Clean strings and handle empty subclasses
    return df.with_columns([
        pl.col('CLASS').cast(pl.String).str.strip_chars(),
        pl.col('SUBCLASS').cast(pl.String).str.strip_chars().fill_null("NORMAL")
    ])


def generate_sdss_mapping(df, overrides):
    """
    Groups all classes. Stars use regex/overrides.
    Galaxies/QSOs use standard prefixing and underscore formatting.
    """
    mapping = defaultdict(list)

    # --- 1. HANDLE GALAXY & QSO (Standard Prefixing) ---
    for cls in ["GALAXY", "QSO"]:
        unique_subs = (
            df.filter(pl.col("CLASS") == cls)
            .select("SUBCLASS")
            .unique()
            .to_series()
            .to_list()
        )

        for raw_name in unique_subs:
            # Replace spaces with underscores and handle empty/NORMAL values
            clean_sub = raw_name.replace(" ", "_")
            if clean_sub == "" or clean_sub == "NORMAL":
                clean_sub = "NORMAL"

            # Result: GALAXY_AGN_BROADLINE or QSO_NORMAL
            final_group_name = f"{cls}_{clean_sub}"
            mapping[final_group_name].append(raw_name)

    # --- 2. HANDLE STARS (Regex + Overrides) ---
    unique_stars = (
        df.filter(pl.col("CLASS") == "STAR")
        .select("SUBCLASS")
        .unique()
        .to_series()
        .to_list()
    )

    for raw_name in unique_stars:
        clean = raw_name.upper()

        # Determine Auto-Generated Name for stars
        if 'WD' in clean:
            auto_name = 'WD'
        elif 'CARBON' in clean:
            auto_name = 'CARBON'
        elif 'CV' in clean:
            auto_name = 'CV'
        elif 'SD' in clean:
            auto_name = 'SUBDWARF_F'
        else:
            # Regex: ^([OBAFGKMLT]\d?)
            match = re.search(r'^([OBAFGKMLT]\d?)', clean)
            auto_name = match.group(1) if match else "UNKNOWN_STAR"

        # Apply Override for stars if exists, else add 'STAR_' prefix to auto_name
        final_group_name = overrides.get(auto_name, f"STAR_{auto_name}")
        mapping[final_group_name].append(raw_name)

    return dict(mapping)


def main():

    # Load custom name overrides
    name_overrides = load_overrides(OVERRIDE_FILE)

    # Process Data
    df = read_sdss_fits(INPUT_FITS)

    # DEBUG: create list with for each CLASS with the unique subclasses in it
    for cls in ["GALAXY", "QSO", "STAR"]:
        unique_subs = (
            df.filter(pl.col("CLASS") == cls)
            .select("SUBCLASS")
            .unique()
            .to_series()
            .to_list()
        )
        print(f"{cls}: {len(unique_subs)} unique subclasses")
        print(unique_subs)

    # Generate Mapping
    final_mapping = generate_sdss_mapping(df, name_overrides)
    final_mapping = {k: sorted(v) for k, v in sorted(final_mapping.items())}

    # Save final JSON
    with open(OUTPUT_MAPPING, 'w') as f:
        json.dump(final_mapping, f, indent=4)

    # Validation & Summary
    print(f"\nMapping complete. Processed {len(final_mapping)} groups.")
    print(f"Results saved to: {OUTPUT_MAPPING}")

    # Verify all subclasses are present
    total_in_mapping = sum(len(v) for v in final_mapping.values())
    total_unique_subclasses = df.select(pl.format("{}_{}", pl.col("CLASS"), pl.col("SUBCLASS")).alias("CLASS_SUB")).unique().height

    if total_in_mapping == total_unique_subclasses:
        print("Consistency Check: [PASSED] (All subclasses accounted for)")
    else:
        print(f"[Warning]: Missing data! Mapping: {total_in_mapping}, Source: {total_unique_subclasses}")


if __name__ == "__main__":
    main()