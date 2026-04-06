import os
import requests
import threading
import polars as pl
import numpy as np
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor


# ==========================================
# CONFIGURATION
# ==========================================
# File paths
CATALOG_FILE = 'assets/spAll-lite-v6_1_3.fits.gz'
OUTPUT_LABELS = 'assets/my_qml_dataset_labels.parquet'
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sdss_spectra')

# Download Settings
SAMPLES_PER_SUBCLASS = 10_000
MAX_WORKERS = 20
TIMEOUT = 30
BASE_URL = "https://data.sdss.org/sas/dr19/spectro/boss/redux"

# Target Classes
TARGET_CATEGORIES = {
    'GALAXY': ['NORMAL', 'BROADLINE', 'AGN', 'AGN BROADLINE', 'STARBURST BROADLINE', 'STARBURST', 'STARFORMING', 'STARFORMING BROADLINE'],
    'QSO': ['NORMAL', 'AGN BROADLINE', 'STARBURST BROADLINE', 'STARFORMING', 'BROADLINE', 'STARBURST', 'AGN', 'STARFORMING BROADLINE'],
    'STAR': ['M4II (175588)', 'B3V (29763)', 'A4p (G_37-26)', 'G5Iab: (20123)', 'M2Iab: (36389)', 'B5 (338529)', 'B0.5Ibe... (187459)', 'M3III (44478)', 'A1Iae (12953)', 'M0III (168720)', 'G0.5IV (14214)', 'F8V (30562)', 'F5Ib... (17463)', 'F9IV (136064)', 'K1V... (25329)', 'A9V (154660)', 'K3Ib... (17506)', 'K2III (115136)', 'B6IIIpe (109387)', 'G0Ib (204867)', 'G9IV (100030)', 'F8V (G_243-63)', 'K3III (101673)', 'A1III (225180)', 'B0.5Iae (185859)', 'K1III (18322)', 'F6V (16673)', 'A1m (78209)', 'B2Iaevar (41117)', 'M6III (148783)', 'M2III (169305)', 'Am (78362)', 'G5/G6IVw (26297)', 'Carbon', 'WDhotter', 'F0III (89025)', 'B1Ve (212571)', 'F0Ib (36673)', 'G5III+... (157910)', 'O8e (188001)', 'K3V (32147)', 'WDmagnetic', 'K4Iab: (34255)', 'G9Ib (221861)', 'F6III (61064)', 'A4V (136729)', 'B2Ve (164284)', 'M4.5:III (123657)', 'sd:F0 (G_84-29)', 'K0V (10780)', 'A2II (39866)', 'K1IVa (142091)', 'M5Iab: (197812)', 'A5Ia (17378)', 'B2.5V (175426)', 'M7IIIevar (177940)', 'F0IV (81937)', 'G1V (95128)', 'O8/O9 (167771)', 'G3Ib (58526)', 'F2III (89254)', 'B8III (220575)', 'K5 (G_19-24)', 'B5III (209419)', 'K0IIIa (57669)', 'K5III (111335)', 'CalciumWD', 'B5V (173087)', 'A4V (97603)', 'B6IV (174959)', 'K5Ve (118100)', 'B9III (15318)', 'F8Ibvar (45412)', 'B0IVe (5394)', 'M8IIIe (84748)', 'sdF3 (140283)', 'B2.5Ve (187811)', 'A3Iae (223385)', 'G8V (75732)', 'O9.5Iae (30614)', 'B2Vne (202904)', 'K5 (110281)', 'G0 (G_101-29)', 'K3p (165195)', 'B3Ve (25940)', 'A5II (34578)', 'F6Iab: (187929)', 'B3II (175156)', 'B2Vne (58343)', 'K0IV (191026)', 'WDcooler', 'B2III (35468)', 'A0 (19510)', 'G4V (32923)', 'A0IVn (25642)', 'K5III (120933)', 'G0Va (143761)', 'B2IV-V (176819)', 'G0 (63791)', 'B5Ib (191243)', 'K3Iab: (4817)', 'A4 (G_165-39)', 'F6II (61295)', 'B7IVe (209409)', 'CV', 'M1 (204445)', 'A1V (95418)', 'K4III (136726)', 'B9 (105262)', 'Ldwarf', 'M5III (221615)', 'F0V (90277)', 'G8V (101501)', 'B9Vn (177756)', 'A8V (155514)', 'A2Ia (14489)', 'B9.5V+... (37269)', 'B3Ib/II (51309)', 'B8IV (171301)', 'F3/F5V (30743)', 'A6IV (28527)', 'F0II (25291)', 'B8Ib (208501)', 'F2V (33256)']
}

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ==========================================
# CATALOG PREPARATION FUNCTIONS
# ==========================================

def prepare_catalog(file_path):
    """Reads FITS, cleans byte-strings, and expands spectral features."""
    print(f"1. Loading {file_path}...")
    astropy_table = Table.read(file_path, hdu=1)

    print("2. Converting to Polars and cleaning strings...")
    data_dict = {}
    for col in astropy_table.colnames:
        arr = np.array(astropy_table[col])
        # Convert Big-Endian (FITS) to Native (Polars)
        if hasattr(arr, 'dtype') and arr.dtype.byteorder == '>':
            arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
        data_dict[col.upper()] = arr

    df = pl.DataFrame(data_dict)

    # Clean byte-string artifacts and whitespace
    df = df.with_columns([
        pl.col(c).cast(pl.String).str.strip_chars().str.replace_all(r"b'|'", "")
        for c in ['CLASS', 'SUBCLASS', 'RUN2D', 'SPEC_FILE']
    ])

    # Fill empty subclasses with 'NORMAL'
    df = df.with_columns(
        pl.when(pl.col('SUBCLASS') == "").then(pl.lit("NORMAL")).otherwise(pl.col('SUBCLASS')).alias('SUBCLASS')
    )

    # Filter for reliable data -> only with no redshift warnings
    df = df.filter(pl.col('ZWARNING') == 0)

    return df


def sample_data(df, targets, samples_per_class):
    """Filters target subclasses and performs random sampling."""
    print("3. Sampling subclasses...")
    selected_chunks = []

    for primary, subclasses in targets.items():
        for sub in subclasses:
            subset = df.filter((pl.col('CLASS') == primary) & (pl.col('SUBCLASS') == sub))

            count = subset.height
            if count == 0:
                print(f"   [SKIP]: No samples found for {primary} - {sub}")
                continue

            n = min(count, samples_per_class)
            selected_chunks.append(subset.sample(n=n, seed=42))

    final_df = pl.concat(selected_chunks)
    print(f"Total samples selected: {final_df.height}")
    return final_df


def finalize_columns(df):
    """Expands SPECTROFLUX array and selects final ML features."""
    print("4. Formatting columns for Machine Learning...")

    # Identify standard columns to keep
    cols_to_keep = [
        'CLASS', 'SUBCLASS', 'Z', 'VDISP', 'Z_ERR', 'ZWARNING',
        'SN_MEDIAN_ALL', 'SPEC_FILE', 'FIELD', 'MJD', 'CATALOGID',
        'TARGET_INDEX', 'FIBERID', 'RUN2D', 'PLATE'
    ]

    # Expand the 5-band SPECTROFLUX (u, g, r, i, z)
    if 'SPECTROFLUX' in df.columns:
        bands = ['U', 'G', 'R', 'I', 'Z']
        df = df.with_columns([
            pl.col("SPECTROFLUX").arr.get(i).alias(f"SPECTROFLUX_{b}")
            for i, b in enumerate(bands)
        ])
        cols_to_keep.extend([f"SPECTROFLUX_{b}" for b in bands])

    return df.select([c for c in cols_to_keep if c in df.columns])


# ==========================================
# DOWNLOAD MANAGEMENT
# ==========================================

class SDSSDownloader:
    def __init__(self, df):
        self.df = df
        self.thread_local = threading.local()
        self.completed = 0
        self.total = df.height

    def get_session(self):
        """Maintains a persistent HTTP connection per thread for speed."""
        if not hasattr(self.thread_local, "session"):
            self.thread_local.session = requests.Session()
            self.thread_local.session.headers.update({
                'User-Agent': 'SDSS-ML-Downloader/2.0 (High-Speed-Session)'
            })
        return self.thread_local.session

    def build_url(self, row):
        """Parses filename to build correct SDSS-V/Legacy path."""
        spec_file = row['SPEC_FILE']
        run2d = row['RUN2D'] or 'v6_1_3'

        try:
            parts = spec_file.split('-')
            field_folder = parts[1]  # Preserves leading zeros
            mjd_folder = parts[2]
            return f"{BASE_URL}/{run2d}/spectra/lite/{field_folder}/{mjd_folder}/{spec_file}"
        except (IndexError, AttributeError):
            return None

    def fetch_file(self, row):
        """Downloads a single file, handling errors and reporting status."""
        url = self.build_url(row)
        path = os.path.join(DOWNLOAD_DIR, row['SPEC_FILE'])

        if not url: return f"Failed (Malformed): {row['SPEC_FILE']}"
        if os.path.exists(path): return None  # Changed to None to keep progress clean

        session = self.get_session()
        try:
            with session.get(url, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with open(path, 'wb') as f:
                    f.write(r.content)

            # Simple progress update inside the thread
            self.completed += 1
            if self.completed % 50 == 0 or self.completed == self.total:
                pct = (self.completed / self.total) * 100
                print(f"   Progress: {self.completed}/{self.total} ({pct:.1f}%)")

            return f"Success: {row['SPEC_FILE']}"
        except Exception:
            return f"Failed: {row['SPEC_FILE']}"

    def run(self, max_passes=25):
        """Repeatedly attempts to download missing files until complete."""
        import time

        for pass_num in range(1, max_passes + 1):
            # Identify missing files
            all_tasks = self.df.to_dicts()
            missing_tasks = [
                row for row in all_tasks
                if not os.path.exists(os.path.join(DOWNLOAD_DIR, row['SPEC_FILE']))
            ]

            if not missing_tasks:
                print(f"\n[DONE] All {self.total} files verified on disk.")
                return

            print(f"\nPass {pass_num}/{max_passes}: Attempting {len(missing_tasks)} missing files...")

            self.completed = self.total - len(missing_tasks)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # We use list() to block until the pass is done
                results = list(executor.map(self.fetch_file, missing_tasks))

            # Count how many actually succeeded this pass
            successes = sum(1 for r in results if r and r.startswith("Success"))
            if successes == 0 and pass_num < max_passes:
                print("No new files downloaded this pass. Waiting 5s for server cool-down...")
                time.sleep(5)

        # Final tally check
        remaining = sum(1 for row in all_tasks if not os.path.exists(os.path.join(DOWNLOAD_DIR, row['SPEC_FILE'])))
        if remaining > 0:
            print(f"\nFinished with {remaining} files still missing after {max_passes} passes.")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Phase 1: Prepare Metadata
    raw_df = prepare_catalog(CATALOG_FILE)
    sampled_df = sample_data(raw_df, TARGET_CATEGORIES, SAMPLES_PER_SUBCLASS)
    final_labels_df = finalize_columns(sampled_df)

    # Phase 2: Save Labels
    final_labels_df.write_parquet(OUTPUT_LABELS)
    print(f"Metadata saved to {OUTPUT_LABELS}")

    # Phase 3: Download
    downloader = SDSSDownloader(final_labels_df)
    downloader.run(max_passes=25)

    print("\nTry Download once again to catch any missed files...")
    downloader.run()  # Run a second time to catch any missed files

    print("aaaand one last time to be sure...")
    downloader.run()  # Final run to catch any stragglers

    print("\nAll tasks completed successfully.")