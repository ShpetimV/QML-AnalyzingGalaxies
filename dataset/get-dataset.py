import pandas as pd
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

# 1. Load HuggingFace dataset
print("Loading HuggingFace dataset...")
ds = load_dataset("MultimodalUniverse/sdss", cache_dir="./data")
split_name = 'train'

# 2. Load CasJobs labels
print("Loading CasJobs labels...")
labels_df = pd.read_csv("sdss-data-full-classes.csv")
labels_df['specObjID'] = labels_df['specObjID'].astype(str).str.strip()

# 3. Process and write in chunks
CHUNK_SIZE = 5000
output_path = "./sdss_merged_full.parquet"
writer = None

print(f"\nProcessing in chunks of {CHUNK_SIZE}...")
for chunk_start in range(0, len(ds[split_name]), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(ds[split_name]))
    print(f"  Chunk {chunk_start:,} → {chunk_end:,}...")

    # Grab only this chunk from HF cache
    chunk = ds[split_name].select(range(chunk_start, chunk_end))

    # Build DataFrame for this chunk
    rows = []
    for example in chunk:
        clean_id = "".join(filter(str.isdigit, str(example['object_id'])))
        rows.append({
            'object_id':      example['object_id'],
            'object_id_clean': clean_id,
            'Z':              example['Z'],
            'Z_ERR':          example['Z_ERR'],
            'ZWARNING':       example['ZWARNING'],
            'VDISP':          example['VDISP'],
            'VDISP_ERR':      example['VDISP_ERR'],
            'SPECTROFLUX_U':  example['SPECTROFLUX_U'],
            'SPECTROFLUX_G':  example['SPECTROFLUX_G'],
            'SPECTROFLUX_R':  example['SPECTROFLUX_R'],
            'SPECTROFLUX_I':  example['SPECTROFLUX_I'],
            'SPECTROFLUX_Z':  example['SPECTROFLUX_Z'],
            'flux':           example['spectrum']['flux'],
            'lambda':         example['spectrum']['lambda'],
            'ivar':           example['spectrum']['ivar'],
            'mask':           example['spectrum']['mask'],
        })

    chunk_df = pd.DataFrame(rows)

    # Merge with labels
    chunk_df = chunk_df.merge(
        labels_df[['specObjID', 'class', 'subClass', 'snMedian']],
        left_on='object_id_clean',
        right_on='specObjID',
        how='left'
    )
    chunk_df.drop(columns=['object_id_clean', 'specObjID'], inplace=True)

    # Write chunk to parquet
    table = pa.Table.from_pandas(chunk_df)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema)
    writer.write_table(table)

if writer:
    writer.close()

print(f"\nDone! Saved to {output_path}")