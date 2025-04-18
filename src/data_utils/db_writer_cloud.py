import os
import sys
import json
import time
import math
import gc

import pandas as pd
import numpy as np

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
R2_OUTPUT_DIR = "integrated_data/viridiplantae_dataset_partitioned_from_json"

DATA_SCHEMA = pa.schema([
    # Basic Info
    pa.field('uniprot_id', pa.large_string(), nullable=True),
    pa.field('sequence', pa.large_string(), nullable=True),
    pa.field('sequence_length', pa.int64(), nullable=True),
    pa.field('organism', pa.large_string(), nullable=True),
    pa.field('taxonomy_id', pa.large_string(), nullable=True),

    # Physicochemical Properties Struct
    pa.field('physicochemical_properties', pa.struct([
        pa.field('aromaticity', pa.float64(), nullable=True),
        pa.field('charge_at_pH_7', pa.float64(), nullable=True),
        pa.field('gravy', pa.float64(), nullable=True),
        pa.field('instability_index', pa.float64(), nullable=True),
        pa.field('isoelectric_point', pa.float64(), nullable=True),
        pa.field('molecular_weight', pa.float64(), nullable=True)
    ]), nullable=True), 

    # Residue Features List
    pa.field('residue_features', pa.list_(pa.struct([
        pa.field('hydrophobicity', pa.float64(), nullable=True),
        pa.field('polarity', pa.float64(), nullable=True),
        pa.field('volume', pa.float64(), nullable=True)
    ])), nullable=True),

    # Structural features list
    pa.field('structural_features', pa.list_(pa.struct([
        pa.field("pdb_id", pa.string(), nullable=True),
        pa.field("pdb_file", pa.string(), nullable=True),
        pa.field("dbref_records", pa.list_(pa.struct([ # List of DBREF structs
            pa.field("chain", pa.string(), nullable=True),
            pa.field("accession", pa.string(), nullable=True),
            pa.field("db_id_code", pa.string(), nullable=True),
            pa.field("pdb_start_res", pa.int64(), nullable=True),
            pa.field("pdb_end_res", pa.int64(), nullable=True),
            pa.field("db_start_res", pa.int64(), nullable=True),
            pa.field("db_end_res", pa.int64(), nullable=True)
        ])), nullable=True),
        pa.field("helix_percentage", pa.float64(), nullable=True),
        pa.field("sheet_percentage", pa.float64(), nullable=True),
        pa.field("coil_percentage", pa.float64(), nullable=True),
        pa.field("total_residues_dssp", pa.int64(), nullable=True), 
        pa.field("dssp_residue_details", pa.list_(pa.struct([ # List of DSSP residue structs
            pa.field("chain", pa.string(), nullable=True),
            pa.field("residue_seq", pa.int64(), nullable=True), # DSSP residue number
            pa.field("residue_icode", pa.string(), nullable=True), # Insertion code
            pa.field("amino_acid", pa.string(), nullable=True), # One-letter code
            pa.field("secondary_structure", pa.string(), nullable=True), # DSSP code (H, E, C etc)
            pa.field("relative_accessibility", pa.float64(), nullable=True),
            pa.field("phi", pa.float64(), nullable=True),
            pa.field("psi", pa.float64(), nullable=True)
        ])), nullable=True),
        pa.field("ca_coordinates", pa.list_(pa.struct([ # List of CA coordinate structs
             pa.field("index", pa.int64(), nullable=True), # Index in the CA list
             pa.field("chain", pa.string(), nullable=True),
             pa.field("residue_seq", pa.int64(), nullable=True),
             pa.field("residue_icode", pa.string(), nullable=True),
             pa.field("x", pa.float64(), nullable=True),
             pa.field("y", pa.float64(), nullable=True),
             pa.field("z", pa.float64(), nullable=True)
        ])), nullable=True),
        # Representing contact map as list of pairs (structs with two indices)
        pa.field("contact_map_indices_ca", pa.list_(pa.struct([
             pa.field("idx1", pa.int64(), nullable=True),
             pa.field("idx2", pa.int64(), nullable=True)
        ])), nullable=True),
        pa.field("processing_error", pa.string(), nullable=True) # Record any errors
    ])), nullable=True), # The outer list itself can be null

    # Domains List
    pa.field('domains', pa.list_(pa.struct([
        pa.field('accession', pa.string(), nullable=True), pa.field('bias', pa.float64(), nullable=True),
        pa.field('description', pa.string(), nullable=True), pa.field('end', pa.int64(), nullable=True),
        pa.field('envelope_end', pa.int64(), nullable=True), pa.field('envelope_start', pa.int64(), nullable=True),
        pa.field('evalue', pa.float64(), nullable=True), pa.field('hmm_end', pa.int64(), nullable=True),
        pa.field('hmm_start', pa.int64(), nullable=True), pa.field('score', pa.float64(), nullable=True),
        pa.field('start', pa.int64(), nullable=True), pa.field('target_name', pa.string(), nullable=True)
    ])), nullable=True),

    # Gelation prediction
    pa.field('gelation', pa.large_string(), nullable=True),

    # Partitioning Column
    pa.field('uniprot_id_prefix', pa.dictionary(pa.int8(), pa.string(), ordered=False), nullable=True)
])

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

# Create storage_options dictionary for Dask/s3fs
storage_options = {
    'key': r2_access_key,
    'secret': r2_secret_key,
    'endpoint_url': r2_endpoint,
    'client_kwargs': {'endpoint_url': r2_endpoint} # Sometimes needed for s3fs
}

# Validate required R2 variables
if not all([r2_access_key, r2_secret_key, r2_bucket_name, r2_endpoint]):
    print("ERROR: Missing Cloudflare R2 credentials/config (KEY, SECRET, BUCKET, ENDPOINT/ACCOUNT_ID) in environment/.env or system environment.")
    sys.exit(1)

print(f"Target R2 Endpoint: {r2_endpoint}")
print(f"Target R2 Bucket:   {r2_bucket_name}")
print(f"Target R2 Path:     {R2_OUTPUT_DIR}")


def preprocess_nans(data):
    """Recursively replaces np.nan with None in nested lists/dicts."""
    if isinstance(data, dict):
        return {k: preprocess_nans(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [preprocess_nans(item) for item in data]
    elif isinstance(data, float) and np.isnan(data):
        return None
    elif pd.isna(data):
         return None
    return data

# --- Main Execution ---
def main():
    print("\n--- Starting JSON to R2 Parquet Upload Script ---")

    start_time = time.time()
    initial_data_len = 0

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
        initial_data_len = len(integrated_data)
        print(f"Successfully loaded {len(integrated_data)} records from JSON.")
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to decode JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read JSON file: {e}")
        sys.exit(1)


    # 2. Convert JSON data to Pandas DataFrame
    print("Converting JSON data to Pandas DataFrame...")
    try:
        data_list = []
        for uniprot_id, data_dict in integrated_data.items():
             if isinstance(data_dict, dict):
                processed_dict = preprocess_nans(data_dict)
                processed_dict['uniprot_id'] = uniprot_id
                data_list.append(processed_dict)
             else:
                 print(f"Warning: Skipping entry for {uniprot_id} as it's not a dictionary.")

        if not data_list:
            print("ERROR: No valid dictionary entries found in the JSON data after processing.")
            sys.exit(1)

        df = pd.DataFrame(data_list)
        df_shape = df.shape

        print(f"Pandas DataFrame created with shape: {df_shape}")
        if 'uniprot_id' not in df.columns:
             print("ERROR: 'uniprot_id' column not found after DataFrame conversion. Check JSON structure and conversion logic.")
             sys.exit(1)
        
        del integrated_data
        del data_list
        gc.collect()

    except MemoryError:
        print("\nERROR: Ran out of memory converting JSON dictionary to Pandas DataFrame.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed during JSON to Pandas DataFrame conversion: {e}")
        sys.exit(1)

    # 3. Add Partitioning Column
    print("Adding partitioning column 'uniprot_id_prefix'...")
    try:
        df['uniprot_id'] = df['uniprot_id'].astype(str)
        df['uniprot_id_prefix'] = df['uniprot_id'].str[0].fillna('?')
        # Convert partitioning column to category for potential efficiency with dictionary type
        df['uniprot_id_prefix'] = df['uniprot_id_prefix'].astype('category')
    except Exception as e:
        print(f"ERROR: Failed to create partitioning column: {e}")
        sys.exit(1)

    # 4. Convert Pandas DataFrame to PyArrow Table
    print("Converting Pandas DataFrame to PyArrow Table with specified schema...")
    try:
        table = pa.Table.from_pandas(df, schema=DATA_SCHEMA, preserve_index=False)
        print("PyArrow Table created successfully")
    except Exception as e:
        print("Pyarrow conversion")
    del df
    gc.collect()


    # 5. Configure PyArrow R2 Filesystem Connection
    print("Configuring R2 filesystem connection...")
    try:
        # Using pandas dataframe
        r2_fs = pafs.S3FileSystem(
            endpoint_override=r2_endpoint,
            access_key=r2_access_key,
            secret_key=r2_secret_key,
            scheme="https" # R2 uses HTTPS
        )
        # Test connection by listing root or target dir
        print("Testing R2 connection (listing bucket root)...")
        print(r2_fs.get_file_info("/"))
    except Exception as e:
        print(f"ERROR: Failed to configure R2 filesystem connection: {e}")
        sys.exit(1)


    # 6. Write PyArrow Table to Partitioned Parquet Dataset on R2
    full_dataset_path = f"{r2_bucket_name}/{R2_OUTPUT_DIR}"
    print(f"Writing partitioned Parquet dataset using PyArrow to: s3://{full_dataset_path}")
    write_start_time = time.time()
    try:
        if table is None:
            print("Error: PyArrow table is None, cannot write")
            sys.exit(1)
        
        pq.write_to_dataset(
            table, 
            root_path = full_dataset_path,
            partition_cols = ['uniprot_id_prefix'],
            schema=DATA_SCHEMA,
            filesystem=r2_fs,
            use_threads=False,
            existing_data_behavior='overwrite_or_ignore'
        )
        write_end_time = time.time()
        print("Data written to Parquet dataset to R2 using PyArrow")
        print(f"PyArrow write time: {write_end_time - write_start_time:.2f} seconds")
    except Exception as e:
            print(f"ERROR: Failed during PyArrow Parquet dataset writing to R2: {e}")
            sys.exit(1)
    finally:
        del table
        gc.collect()

    # 7. Final Summary
    end_time = time.time()
    print("\n--- Upload Summary ---")
    print(f"Input JSON records:      {initial_data_len}")
    print(f"Parquet dataset written: s3://{full_dataset_path}")
    print(f"Total time taken:      {end_time - start_time:.2f} seconds")
    print("------------------------")

if __name__ == "__main__":
    main()