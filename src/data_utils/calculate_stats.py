import os
import argparse
import json
import sys
import time 
import gc

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from dotenv import load_dotenv

# --- Configuration ---
ENV_FILE_PATH = ".env"
# R2 Path for the INPUT dataset (MUST match db_writer_cloud.py output)
DEFAULT_R2_DATASET_DIR = "integrated_data/viridiplantae_dataset_partitioned_from_json"
STATS_OUTPUT_FILE = "mean_std.json" # Output file for dataset.py
GELATION_DOMAINS = ["PF00190", "PF04702", "PF00234"]

# Columns needed from Parquet (ensure these match what's needed by helpers)
COLUMNS_TO_READ = ['uniprot_id', 'sequence_length', 'physicochemical_properties',
                   'structural_features', 'residue_features', 'domains']

# Number of expected features derived from helpers (for clarity)
# Protein: 10 continuous + num_gelation_domains
EXPECTED_NUM_PROTEIN_FEATURES = 10 + len(GELATION_DOMAINS)
# Residue: 6 continuous (hydrophobicity, polarity, volume, acc, phi, psi)
EXPECTED_NUM_RESIDUE_FEATURES = 6

# --- Load Environment Variables & Configure R2 FS ---
print(f"Loading R2 credentials from: {ENV_FILE_PATH}")
if not load_dotenv(dotenv_path=ENV_FILE_PATH):
    print(f"Warning: .env file not found at {ENV_FILE_PATH}. Using system environment variables.")

r2_access_key = os.getenv("CLOUDFARE_ACCESS_KEY")
r2_secret_key = os.getenv("CLOUDFARE_SECRET_KEY")
r2_account_id = os.getenv("CLOUDFARE_ACCOUNT_ID")
r2_bucket_name = os.getenv("CLOUDFARE_BUCKET_NAME")
r2_endpoint = os.getenv("CLOUDFARE_ENDPOINT")

if not r2_endpoint and r2_account_id:
    r2_endpoint = f"https://{r2_account_id}.r2.cloudflarestorage.com"
if not all([r2_access_key, r2_secret_key, r2_bucket_name, r2_endpoint]):
    print("ERROR: Missing Cloudflare R2 credentials/config (KEY, SECRET, BUCKET, ENDPOINT/ACCOUNT_ID) in environment/.env or system environment.")
    sys.exit(1)


# Configure PyArrow R2 Filesystem
try:
    r2_fs = pafs.S3FileSystem(
        endpoint_override=r2_endpoint,
        access_key=r2_access_key,
        secret_key=r2_secret_key,
        scheme="https" # R2 uses HTTPS
    )

    print(f"Testing R2 connection to bucket '{r2_bucket_name}'...")
    r2_fs.get_file_info(f"{r2_bucket_name}/") # Check connectivity to the bucket itself
    print("R2 Filesystem connection successful.")
except Exception as e:
    print(f"ERROR: Failed to configure or connect R2 filesystem for bucket '{r2_bucket_name}': {e}")
    sys.exit(1)

print(f"Using R2 Endpoint: {r2_endpoint}, Bucket: {r2_bucket_name}")


# --- Helper Functions for Feature Extraction ---
def extract_protein_features_numeric(row):
    """
        Extracts numeric protein features from a row (Pandas Series).
        Handles missing data by returning NaNs or default values.
        Returns a NumPy array of fixed size (EXPECTED_NUM_PROTEIN_FEATURES).
    """
    try:
        phy_prop = row.get('physicochemical_properties', {}) or {}
        struct_list = row.get('structural_features', []) or []
        # Get first structure if list is not empty and first item is a dict
        struct = struct_list[0] if struct_list and isinstance(struct_list[0], dict) else {}
        domains = row.get('domains', []) or []

        # Extract continuous features, defaulting to np.nan if missing or not finite
        def safe_float(value, default=np.nan):
            try:
                # Check if value is already a float/int and finite
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
                # Attempt conversion if not already suitable type
                if value is not None:
                    f_val = float(value)
                    return f_val if np.isfinite(f_val) else default
                return default # Return default if value is None
            except (TypeError, ValueError):
                return default


        prot_cont = [
            safe_float(row.get('sequence_length', 0)), # Use get for top-level too
            safe_float(phy_prop.get('molecular_weight')),
            safe_float(phy_prop.get('aromaticity')),
            safe_float(phy_prop.get('instability_index')),
            safe_float(phy_prop.get('isoelectric_point')),
            safe_float(phy_prop.get('gravy')),
            safe_float(phy_prop.get('charge_at_pH_7')),
            safe_float(struct.get('helix_percentage')),
            safe_float(struct.get('sheet_percentage')),
            safe_float(struct.get('coil_percentage'))
        ]

        # Extract domain flags (categorical binary)
        domain_flags = [1.0 if any(isinstance(d, dict) and d.get('accession') == gd for d in domains) else 0.0
                        for gd in GELATION_DOMAINS]

        # Combine and return as float64 numpy array
        feature_vector = np.array(prot_cont + domain_flags, dtype=np.float64)

        # Final shape check
        if feature_vector.shape[0] != EXPECTED_NUM_PROTEIN_FEATURES:
             print(f"Warning: Protein feature shape mismatch. Expected {EXPECTED_NUM_PROTEIN_FEATURES}, got {feature_vector.shape[0]}. Filling with NaNs.")
             return np.full(EXPECTED_NUM_PROTEIN_FEATURES, np.nan, dtype=np.float64)

        return feature_vector

    except Exception as e:
        # Broad exception catch to ensure Dask doesn't fail on a single bad row
        return np.full(EXPECTED_NUM_PROTEIN_FEATURES, np.nan, dtype=np.float64)

def extract_residue_features_numeric(row):
    """
    Extracts numeric residue features as a list of lists for a single protein row.
    Each inner list contains EXPECTED_NUM_RESIDUE_FEATURES features.
    Handles missing/invalid data within residues.
    """
    residue_data = []
    try:
        seq_len = row.get('sequence_length', 0)
        # Handle case where sequence length is missing or zero
        if not isinstance(seq_len, int) or seq_len <= 0:
            return []

        # Use .get() with default empty list for safety
        seq_res_feats = row.get('residue_features', [])
        struct_list = row.get('structural_features', [])
        struct = struct_list[0] if struct_list and isinstance(struct_list[0], dict) else {}
        struct_res_details = struct.get('dssp_residue_details', [])

        for i in range(seq_len):
            # Get features for residue i
            res_f = seq_res_feats[i] if i < len(seq_res_feats) and isinstance(seq_res_feats[i], dict) else {}
            # Use dssp_residue_details from struct
            res_d = struct_res_details[i] if i < len(struct_res_details) and isinstance(struct_res_details[i], dict) else {}

            # Extract continuous features, handling None/NaN/TypeErrors, default to np.nan
            def safe_float(value, default=np.nan):
                try:
                    if isinstance(value, (int, float)) and np.isfinite(value):
                       return float(value)
                    if value is not None:
                       f_val = float(value)
                       return f_val if np.isfinite(f_val) else default
                    return default
                except (TypeError, ValueError):
                    return default

            features = [
                safe_float(res_f.get('hydrophobicity')),
                safe_float(res_f.get('polarity')),
                safe_float(res_f.get('volume')),
                safe_float(res_d.get('relative_accessibility')),
                safe_float(res_d.get('phi')),
                safe_float(res_d.get('psi'))
            ]

            # Ensure correct number of features extracted for this residue
            if len(features) != EXPECTED_NUM_RESIDUE_FEATURES:
                 residue_data.append([np.nan] * EXPECTED_NUM_RESIDUE_FEATURES)
            else:
                 residue_data.append(features)

        # Return list of lists/arrays, inner is [hydro, polar, vol, acc, phi, psi]
        return residue_data # Each element is a list of features for one residue

    except Exception as e:
        # import traceback  
        # traceback.print_exc()
        return [] 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate protein/residue feature statistics from a Parquet dataset on R2.")
    parser.add_argument('--r2_input_path', type=str, default=DEFAULT_R2_DATASET_DIR,
                        help=f'Path to the INPUT Parquet dataset directory within the R2 bucket. Default: {DEFAULT_R2_DATASET_DIR}')
    parser.add_argument('--output_file', type=str, default=STATS_OUTPUT_FILE,
                        help=f'Path to save the output JSON statistics file. Default: {STATS_OUTPUT_FILE}')
    args = parser.parse_args()

    print("--- Starting Statistics Calculation using Pandas/NumPy ---")
    start_time = time.time()


    full_dataset_path = f"{r2_bucket_name}/{args.r2_input_path}"
    stats = {} # Dictionary to store results
    df = None # Dataframe vaiable
    protein_feature_np_array = None
    residue_features_np_array = None
    protein_features_series = None

    try:
        # Read Entire Parquet Dataset into Pandas DataFrame
        print(f"\nReading Parquet dataset into Pandas DataFrame from: s3://{full_dataset_path}")
        print("This may take significant time and memory depending on dataset size.")
        load_start = time.time()

        try:
            # Use PyArrwo ParquetDataset to read data
            dataset = pq.ParquetDataset(
                full_dataset_path, 
                filesystem=r2_fs
            )
            # Read columns into PyArrow Table, then convert to Pandas
            table = dataset.read(columns=COLUMNS_TO_READ)
            df = table.to_pandas()
            del table, dataset
            gc.collect()
            load_end = time.time()
            print(f"Pandas DataFrame loaded successfully with shape: {df.shape}")
            print(f"Data loading took {load_end - load_start:.2f} seconds.")
            if df.empty:
                 print("ERROR: Loaded DataFrame is empty. Check the input path and data.")
                 sys.exit(1)
        except Exception as e:
            print(f"\nERROR: Failed to load Parquet dataset into Pandas: {e}")
            sys.exit(1)
        

        # Calcualte Protein Stats
        print("Calculating protein level stats")
        protein_start_time = time.time()

        # Apply Extraction function row wise -> Series of numpy arrays
        protein_features_series = df.apply(extract_protein_features_numeric, axis=1)

        protein_features_list = protein_features_series.tolist()
        protein_features_list = [arr for arr in protein_features_list if isinstance(arr, np.ndarray)]
        
        if not protein_features_list:
            print("ERROR: No Valid Protein feature list extracted")
            sys.exit()

        print("Stacking Protein feature list into numpy Array")
        protein_feature_np_array = np.stack(protein_features_list)
        print(f" Protein feature array shape: {protein_feature_np_array.shape}")

        # Compute mean and std using numpy funtions
        print("Computing mean and std")
        with np.errstate(invalid='ignore'):
            protein_mean_computed = np.nanmean(protein_feature_np_array.astype(np.float64), axis=0)
            protein_std_computed = np.nanstd(protein_feature_np_array.astype(np.float64), axis=0)
        
        proteint_end_time = time.time()


        # Compute residue stats
        print("Calculating residue stats")
        residue_start_time = time.time()

        all_residue_features_list = []
        print("Extracting all residue features...")
        extraction_errors = 0
        processed_rows = 0
        total_rows = len(df)

        print("Applying residue extraction function (this can take time)...")
        apply_start = time.time()
        residue_results_series = df.apply(extract_residue_features_numeric, axis=1)
        apply_end = time.time()
        print(f"Residue extraction function applied in {apply_end - apply_start:.2f} seconds.")

        # Now process the results
        print("Processing extracted residue feature lists...")
        for result_list in residue_results_series:
            if result_list:
                all_residue_features_list.extend(result_list)
            else:
                extraction_errors += 1
            processed_rows += 1
            if processed_rows % 5000 == 0: # Progress indicator
                 print(f"    Processed {processed_rows}/{total_rows} proteins' residue results...", end='\r')

        print(f"\nFinished processing residue results for {processed_rows} proteins.") # Newline
        if extraction_errors > 0:
            print(f"Note: Failed to extract any residue features for {extraction_errors} rows.")

        if not all_residue_features_list:
            print("ERROR: No valid residue features were collected from any protein.")
            # Optional: Add more debug info here if needed, e.g., print residue_results_series.head()
            sys.exit(1)
        
        print("Converting List of lists into numpy array")
        print(f"Converting {len(all_residue_features_list)} residue features into numpy array")
        try:
            residue_features_np_array = np.array(all_residue_features_list, dtype=np.float64)
            del all_residue_features_list
            gc.collect()
            print(f"Residue features array shape {residue_features_np_array.shape}")
        except Exception as e:
            print(f"Error: While converting list to np array {e}")
        
        # Compute mean and std using numpy functions
        print("Compute residue mean and std")
        with np.errstate(invalid='ignore'):
            residue_mean_computed = np.nanmean(residue_features_np_array, axis=0)
            residue_std_computed = np.nanstd(residue_features_np_array, axis=0)
        
        residue_end_time = time.time()
        print(f"Residue stats computed in {residue_end_time - residue_start_time:.2f} seconds.")

        
        # Check
        epsilon = 1e-7
        protein_mean_final = np.nan_to_num(protein_mean_computed, nan=0.0)
        protein_std_final = np.nan_to_num(protein_std_computed, nan=0.0)
        residue_mean_final = np.nan_to_num(residue_mean_computed, nan=0.0)
        residue_std_final = np.nan_to_num(residue_std_computed, nan=0.0)

        # Ensure std dev is not zero after nan_to_num
        protein_std_final[protein_std_final < epsilon] = epsilon
        residue_std_final[residue_std_final < epsilon] = epsilon


        stats = {
            # Convert to list for JSON serialization
            'protein_mean': protein_mean_final.tolist(),
            'protein_std': protein_std_final.tolist(),
            'residue_mean': residue_mean_final.tolist(),
            'residue_std': residue_std_final.tolist()
        }

        
        # Saving
        print(f"\n Saving statistics to: {args.output_file}")
        try:
            output_dir = os.path.dirname(args.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(args.output_file, 'w') as f:
                json.dump(stats, f, indent=4)
            
            print("Statistics saved successfully.")
        except Exception as e:
            print(f"Error saving stats: {e}")

    except Exception as e:
        print(f"Error occurred while computing stats: {e}")
    finally:
        del df, protein_features_series, protein_feature_np_array, residue_features_np_array
        gc.collect()
    
    end_time = time.time()
    print("Statistics computation finished")
    print(f"Execution time: {end_time-start_time:.2f} seconds")