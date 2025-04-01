import os
import sys
import json
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from dotenv import load_dotenv

# --- Configuration ---
BASE_DATA_DIR = "../../data"
INPUT_JSON_FILENAME = "integrated_data.json"

# Cloudflare R2 Output Configuration
ENV_FILE_PATH = ".env" # Assumes .env file in the script's directory
# Target path WITHIN the R2 bucket
R2_OUTPUT_DIR = "integrated_data/viridiplantae_dataset_partitioned_from_json" # Example path

# --- Path Setup ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_abs_path = os.path.abspath(os.path.join(script_dir, BASE_DATA_DIR))
    input_json_path = os.path.join(base_abs_path, INPUT_JSON_FILENAME)
except Exception as e:
    print(f"Error determining script directory or paths: {e}")
    sys.exit(1)

# --- Load Environment Variables for R2 ---
print(f"Loading R2 credentials from: {ENV_FILE_PATH}")
if not load_dotenv(dotenv_path=ENV_FILE_PATH):
    print(f"Warning: .env file not found at {ENV_FILE_PATH}. Using system environment variables.")
    # Check if running in an environment where .env is needed but missing
    if not all(os.getenv(k) for k in ["CLOUDFARE_ACCESS_KEY", "CLOUDFARE_SECRET_KEY", "CLOUDFARE_BUCKET_NAME"]) or \
       not (os.getenv("CLOUDFARE_ACCOUNT_ID") or os.getenv("CLOUDFARE_ENDPOINT")):
        print("Warning: Critical R2 environment variables might be missing.")

r2_access_key = os.getenv("CLOUDFARE_ACCESS_KEY")
r2_secret_key = os.getenv("CLOUDFARE_SECRET_KEY")
r2_account_id = os.getenv("CLOUDFARE_ACCOUNT_ID")
r2_bucket_name = os.getenv("CLOUDFARE_BUCKET_NAME")
r2_endpoint = os.getenv("CLOUDFARE_ENDPOINT") # Explicit endpoint overrides account ID

# Construct endpoint from account ID if endpoint not explicitly set
if not r2_endpoint and r2_account_id:
    r2_endpoint = f"https://{r2_account_id}.r2.cloudflarestorage.com"
elif not r2_endpoint and not r2_account_id:
     print("ERROR: Missing Cloudflare R2 endpoint or account ID (set CLOUDFARE_ENDPOINT or CLOUDFARE_ACCOUNT_ID).")
     sys.exit(1)


# Validate required R2 variables
if not all([r2_access_key, r2_secret_key, r2_bucket_name, r2_endpoint]):
    print("ERROR: Missing Cloudflare R2 credentials/config (KEY, SECRET, BUCKET, ENDPOINT/ACCOUNT_ID) in environment/.env or system environment.")
    sys.exit(1)

print(f"Target R2 Endpoint: {r2_endpoint}")
print(f"Target R2 Bucket:   {r2_bucket_name}")
print(f"Target R2 Path:     {R2_OUTPUT_DIR}")


# --- Main Execution ---
def main():
    print("\n--- Starting JSON to R2 Parquet Upload Script ---")
    start_time = time.time()

    # 1. Check and Load Input JSON
    if not os.path.exists(input_json_path):
        print(f"ERROR: Input JSON file not found: {input_json_path}")
        sys.exit(1)

    print(f"Loading data from {input_json_path}...")
    try:
        with open(input_json_path, 'r') as f:
            integrated_data = json.load(f)
        if not integrated_data:
            print("ERROR: JSON file is empty or contains no data.")
            sys.exit(1)
        print(f"Successfully loaded {len(integrated_data)} records from JSON.")
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to decode JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read JSON file: {e}")
        sys.exit(1)

    # 2. Convert JSON data to Pandas DataFrame
    # Assumes JSON structure is {"uniprot_id": {feature_dict}}
    print("Converting JSON data to Pandas DataFrame...")
    try:
        df = pd.DataFrame.from_dict(integrated_data, orient='index')
        # Reset index to make 'uniprot_id' a column
        df = df.reset_index().rename(columns={'index': 'uniprot_id'})
        print(f"DataFrame created with shape: {df.shape}")
        # Basic validation
        if 'uniprot_id' not in df.columns:
             print("ERROR: 'uniprot_id' column not found after DataFrame conversion. Check JSON structure.")
             sys.exit(1)
    except MemoryError:
        print("\nERROR: Ran out of memory converting JSON to DataFrame.")
        print("The JSON file might be too large for this machine's RAM.")
        print("Consider processing the JSON in chunks or using a machine with more RAM.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed during JSON to DataFrame conversion: {e}")
        sys.exit(1)

    # 3. Add Partitioning Column
    print("Adding partitioning column 'uniprot_id_prefix'...")
    try:
        # Ensure uniprot_id is string type before slicing
        df['uniprot_id'] = df['uniprot_id'].astype(str)
        df['uniprot_id_prefix'] = df['uniprot_id'].str[0].fillna('?') # Use '?' for empty/null IDs
    except Exception as e:
         print(f"ERROR: Failed to create partitioning column: {e}")
         sys.exit(1)

    # 4. Convert DataFrame to PyArrow Table
    print("Converting DataFrame to PyArrow Table...")
    try:
        # preserve_index=False because we already reset the index
        # PyArrow might issue warnings about converting complex types (like lists of dicts)
        # - these are often fine but monitor if issues arise during writing/reading
        table = pa.Table.from_pandas(df, preserve_index=False)
        print("PyArrow Table created successfully.")
        # print("Schema:", table.schema) # Uncomment to inspect inferred schema
    except Exception as e:
        print(f"ERROR: Failed during DataFrame to PyArrow Table conversion: {e}")
        sys.exit(1)

    # 5. Configure R2 Filesystem Connection
    print("Configuring R2 filesystem connection...")
    try:
        r2_fs = pafs.S3FileSystem(
            endpoint_override=r2_endpoint,
            access_key=r2_access_key,
            secret_key=r2_secret_key,
            scheme="https" # R2 uses HTTPS
            # region parameter usually not needed for R2 unless specified otherwise
        )
        # Test connection by listing root or target dir (can be slow)
        print("Testing R2 connection (listing bucket root)...")
        print(r2_fs.get_file_info("/"))
    except Exception as e:
        print(f"ERROR: Failed to configure R2 filesystem connection: {e}")
        sys.exit(1)

    # 6. Write Parquet Dataset to R2
    full_dataset_uri = f"{r2_bucket_name}/{R2_OUTPUT_DIR}"
    print(f"Writing partitioned Parquet dataset to: {full_dataset_uri}")
    write_start_time = time.time()
    try:
        pq.write_to_dataset(
            table,
            root_path=full_dataset_uri,
            partition_cols=['uniprot_id_prefix'],
            filesystem=r2_fs,
            use_threads=True, # Enable multi-threaded writing
            existing_data_behavior='overwrite_or_ignore' # Overwrite if exists, good for reruns
        )
        write_end_time = time.time()
        print("Successfully wrote Parquet dataset to R2.")
        print(f"R2 write time: {write_end_time - write_start_time:.2f} seconds")

    except Exception as e:
        print(f"ERROR: Failed during Parquet dataset writing to R2: {e}")
        sys.exit(1)

    # 7. Final Summary
    end_time = time.time()
    print("\n--- Upload Summary ---")
    print(f"Input JSON records:      {len(integrated_data)}")
    print(f"DataFrame dimensions:    {df.shape}")
    print(f"Parquet dataset written: {full_dataset_uri}")
    print(f"Total time taken:        {end_time - start_time:.2f} seconds")
    print("------------------------")

if __name__ == "__main__":
    main()