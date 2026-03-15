import requests
import pandas as pd
from io import StringIO
from datasets import load_dataset
from tqdm import tqdm

# 1. Load your downloaded dataset
print("Loading Hugging Face dataset...")
ds = load_dataset("MultimodalUniverse/sdss", cache_dir="./data")

# Let's assume we are working with the 'train' split.
split_name = 'train'

# 2. Extract and clean all unique IDs (<- THIS IS THE FIXED PART!)
print("Extracting unique IDs...")
unique_ids = set()
for raw_id in ds[split_name]['object_id']:
    # Convert whatever it is into a standard string first
    raw_id_str = str(raw_id)

    # Filter out everything except the actual numbers (bye bye "b" and "'")
    clean_id_str = "".join(filter(str.isdigit, raw_id_str))

    # Convert the clean string of digits into an integer
    if clean_id_str:
        unique_ids.add(int(clean_id_str))

unique_ids = list(unique_ids)
print(f"Found {len(unique_ids)} unique objects to classify.")

# 3. Fetch labels in batches
BATCH_SIZE = 50  # 50 IDs creates a safe URL length
url = 'http://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch'
id_to_class_dict = {}

print("Fetching labels from SDSS API in batches...")
for i in tqdm(range(0, len(unique_ids), BATCH_SIZE)):
    batch = unique_ids[i: i + BATCH_SIZE]

    id_str = ",".join(map(str, batch))
    sql_query = f"SELECT specObjID, class FROM SpecObj WHERE specObjID IN ({id_str})"

    try:
        # BACK TO GET REQUESTS! We pass the data via 'params' instead of 'data'
        response = requests.get(url, params={'cmd': sql_query, 'format': 'csv'})

        if response.status_code == 200:
            if "ERROR" in response.text.upper() or "EXCEPTION" in response.text.upper():
                print(f"\nSDSS Server Error at index {i}: {response.text.strip()}")
                break

            batch_df = pd.read_csv(StringIO(response.text), skiprows=1)
            for _, row in batch_df.iterrows():
                id_to_class_dict[int(row['specObjID'])] = row['class']
        else:
            print(f"\nHTTP Error {response.status_code} at index {i}:")
            print(response.text[:200])
            break

    except Exception as e:
        print(f"\nConnection failed: {e}")
        break

# 4. Map the labels back to the Hugging Face dataset
print("\nMapping labels to create the new dataset...")


def add_label(example):
    raw_id_str = str(example['object_id'])
    clean_id_str = "".join(filter(str.isdigit, raw_id_str))

    if clean_id_str:
        clean_id = int(clean_id_str)
        example['class'] = id_to_class_dict.get(clean_id, 'UNKNOWN')
    else:
        example['class'] = 'UNKNOWN'

    return example


# This creates a brand new dataset object with your extra column!
labeled_ds = ds[split_name].map(add_label)

# 5. Verify the results!
print("\nSuccess! Here is a preview of your new labeled dataset:")
print(labeled_ds[0])

# Main
def checkDataset():
    from collections import Counter

    # 1. Print a pretty, readable table of the first 10 rows
    print("\n--- First 10 Rows (Preview) ---")
    # We grab the first 10 rows and convert just the ID and Class columns to a DataFrame
    preview_df = pd.DataFrame(labeled_ds[:10])

    # If your dataset has other columns like 'flux', we just print these two to keep it clean
    if 'object_id' in preview_df.columns and 'class' in preview_df.columns:
        print(preview_df[['object_id', 'class']])
    else:
        print(preview_df)

        # 2. Check the overall distribution of labels
    print("\n--- Label Distribution ---")
    # This will count exactly how many Stars, Galaxies, Quasars, and Unknowns you got
    label_counts = Counter(labeled_ds['class'])

    for label, count in label_counts.items():
        print(f"{label}: {count} objects")

if __name__ == "__main__":
    checkDataset()

