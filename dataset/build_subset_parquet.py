from pathlib import Path
import json
import numpy as np
import polars as pl
from astropy.table import Table

HERE = Path(__file__).resolve().parent
DEFAULT_FITS = HERE / "assets" / "spAll-lite-v6_1_3.fits.gz"
DEFAULT_MAPPING = HERE / "assets" / "subclass_merge_groups.json"
DEFAULT_LABELS = HERE / "assets" / "my_qml_dataset_labels.parquet"

# -----------------------
# Configuration
# -----------------------
GROUPS = ["STAR_A0", "STAR_F5", "STAR_F9", "STAR_G2", "STAR_K1", "STAR_K3", "STAR_K5", "GALAXY_NORMAL", "GALAXY_AGN",
          "GALAXY_STARBURST", "GALAXY_STARFORMING", "QSO_NORMAL", "QSO_BROADLINE"]

OUT_PATH = HERE / "subset_GasNetII_replica.parquet"
MAPPING_JSON = DEFAULT_MAPPING
FITS_PATH = DEFAULT_FITS
LABELS_PARQUET = DEFAULT_LABELS
MAX_PER_GROUP = None
SEED = 42
FILTER_ZWARNING = True
INCLUDE_SPECTROFLUX = False


def build_reverse_map(mapping_json_path):
    with open(mapping_json_path, 'r') as f:
        mapping = json.load(f)

    reverse_map = {}
    for group_name, subclasses in mapping.items():
        # Determine prefix by group name to avoid collisions
        if group_name.startswith("GALAXY_"):
            prefix = "GALAXY"
        elif group_name.startswith("QSO_"):
            prefix = "QSO"
        else:
            prefix = "STAR"

        for sub in subclasses:
            if sub == "":
                sub = "NORMAL"
            reverse_map[f"{prefix}_{sub}"] = group_name

    return reverse_map


def load_labels_from_fits(fits_path: Path, filter_zwarning: bool = False) -> pl.DataFrame:
    print(f"Reading FITS catalog from {fits_path}...")
    tab = Table.read(str(fits_path), hdu=1)
    data = {}
    for col in tab.colnames:
        arr = np.array(tab[col])
        if hasattr(arr, 'dtype') and getattr(arr.dtype, 'byteorder', '=') == '>':
            arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
        data[col.upper()] = arr

    df = pl.DataFrame(data)

    text_cols = [c for c in ['CLASS', 'SUBCLASS', 'RUN2D', 'SPEC_FILE'] if c in df.columns]
    if text_cols:
        df = df.with_columns([
            pl.col(c).cast(pl.Utf8).str.strip_chars().str.replace_all(r"b'|'", "")
            for c in text_cols
        ])

    if 'SUBCLASS' in df.columns:
        df = df.with_columns(
            pl.when(pl.col('SUBCLASS') == "").then(pl.lit('NORMAL')).otherwise(pl.col('SUBCLASS')).alias('SUBCLASS')
        )

    if filter_zwarning and 'ZWARNING' in df.columns:
        df = df.filter(pl.col('ZWARNING') == 0)

    if 'SN_MEDIAN_ALL' in df.columns:
        df = df.filter(pl.col('SN_MEDIAN_ALL') >= 30)

    return df


def load_labels_from_parquet(parquet_path: Path) -> pl.DataFrame:
    print(f"Reading labels parquet from {parquet_path}...")
    return pl.read_parquet(parquet_path)


def build_subset(groups, out_path: Path, mapping_json=MAPPING_JSON, fits_path=FITS_PATH,
                 labels_parquet=LABELS_PARQUET, max_per_group=MAX_PER_GROUP, seed=SEED, filter_zwarning=FILTER_ZWARNING,
                 include_spectroflux=INCLUDE_SPECTROFLUX):
    """Build and write subset parquet. Returns the output DataFrame."""
    reverse_map = build_reverse_map(mapping_json)

    if Path(fits_path).exists():
        df = load_labels_from_fits(Path(fits_path), filter_zwarning=filter_zwarning)
    elif Path(labels_parquet).exists():
        df = load_labels_from_parquet(Path(labels_parquet))
    else:
        raise FileNotFoundError("Neither FITS catalog nor labels parquet were found.")

    if 'CLASS' not in df.columns or 'SUBCLASS' not in df.columns:
        raise KeyError('Input labels must contain CLASS and SUBCLASS columns')

    df = df.with_columns(
        pl.format("{}_{}", pl.col('CLASS'), pl.col('SUBCLASS')).alias('LOOKUP_ID')
    )

    df = df.with_columns(
        pl.col('LOOKUP_ID').replace_strict(reverse_map, default=None).alias('MERGED_GROUP')
    )

    df = df.with_columns(
        pl.col('MERGED_GROUP').alias('FINAL_CLASS')
    )

    print('\nRequested merged groups and available subclasses (found in source):')
    for grp in groups:
        grp_df = df.filter(pl.col('MERGED_GROUP') == grp)
        if grp_df.height == 0:
            print(f"  {grp}: NONE found in source mapping/catalog")
            continue

        # print the first 20 subclasses with counts for this group
        sub_counts = grp_df.group_by('SUBCLASS').agg(pl.len()).sort('len', descending=True)
        subs_list = [f"{r['SUBCLASS']} ({r['len']})" for r in sub_counts.iter_rows(named=True)]
        print(
            f"  {grp}: {len(subs_list)} subclass(es) -> {', '.join(subs_list[:20])}{'...' if len(subs_list) > 20 else ''}")

    groups = set(groups)
    df_filtered = df.filter(pl.col('MERGED_GROUP').is_in(list(groups)))

    if df_filtered.height == 0:
        print('No samples found for the requested groups. Exiting.')
        return None

    if max_per_group is not None:
        df_filtered = (
            df_filtered
            .sample(fraction=1.0, shuffle=True, seed=seed)
            .filter(pl.int_range(0, pl.len()).over('MERGED_GROUP') < max_per_group)
        )

    cols_keep = [c for c in
                 ['CLASS', 'SUBCLASS', 'MERGED_GROUP', 'FINAL_CLASS', 'SN_MEDIAN_ALL', 'SPEC_FILE', 'FIELD', 'MJD',
                  'CATALOGID', 'PLATE', 'FIBERID'] if c in df_filtered.columns]
    if include_spectroflux and 'SPECTROFLUX' in df_filtered.columns:
        bands = ['U', 'G', 'R', 'I', 'Z']
        for i, b in enumerate(bands):
            if 'SPECTROFLUX' in df_filtered.columns:
                df_filtered = df_filtered.with_columns(pl.col('SPECTROFLUX').arr.get(i).alias(f'SPECTROFLUX_{b}'))
        cols_keep += [f'SPECTROFLUX_{b}' for b in bands]

    out_df = df_filtered.select(cols_keep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    print(f'Wrote subset parquet: {out_path} ({out_df.height} rows)')
    return out_df


def _print_report(df: pl.DataFrame):
    if df is None or df.height == 0:
        print('No dataframe available for report.')
        return
    print('\nSUBSET REPORT')
    print('--------------')
    if 'FINAL_CLASS' in df.columns:
        counts = df.group_by('FINAL_CLASS').len().sort('len', descending=True)
        for row in counts.iter_rows(named=True):
            print(f"{row['FINAL_CLASS']:<30} {row['len']:>6} samples")
    elif 'MERGED_GROUP' in df.columns:
        counts = df.group_by('MERGED_GROUP').len().sort('len', descending=True)
        for row in counts.iter_rows(named=True):
            print(f"{row['MERGED_GROUP']:<30} {row['len']:>6} samples")
    else:
        print('Neither FINAL_CLASS nor MERGED_GROUP present; cannot report per-group counts.')


if __name__ == '__main__':
    out_df = build_subset(GROUPS, OUT_PATH, mapping_json=MAPPING_JSON, fits_path=FITS_PATH,
                          labels_parquet=LABELS_PARQUET, max_per_group=MAX_PER_GROUP, seed=SEED,
                          filter_zwarning=FILTER_ZWARNING, include_spectroflux=INCLUDE_SPECTROFLUX)
    _print_report(out_df)
