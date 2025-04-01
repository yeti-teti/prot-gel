import os
import json
import sys
import time 

import numpy as np
import pandas as pd

import dask.dataframe as dd
import dask.array as da

import pyarrow.fs as pafs
from dask.distributed import Client, LocalCluster

from dotenv import load_dotenv

# --- Configuration ---
ENV_FILE_PATH = ".env"
# R2 Path for the INPUT dataset (MUST match db_writer_cloud.py output)
R2_DATASET_DIR = "integrated_data/viridiplantae_dataset_partitioned_from_json"
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

# Prepare storage options for Dask
storage_options = {'key': r2_access_key, 'secret': r2_secret_key, 'endpoint_url': r2_endpoint}
print(f"Using R2 Endpoint: {r2_endpoint}, Bucket: {r2_bucket_name}")


# --- Helper Functions for Feature Extraction (to be used with Dask) ---
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
                f_val = float(value)
                return f_val if np.isfinite(f_val) else default
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
        seq_res_feats = row.get('residue_features', []) or []
        struct_list = row.get('structural_features', []) or []
        struct = struct_list[0] if struct_list and isinstance(struct_list[0], dict) else {}
        struct_res_details = struct.get('dssp_residue_details', []) or [] # Corrected key based on db_writer.py output

        for i in range(seq_len):
            # Get features for residue i, handling potential index errors or type issues
            res_f = seq_res_feats[i] if i < len(seq_res_feats) and isinstance(seq_res_feats[i], dict) else {}
            # Use dssp_residue_details from struct
            res_d = struct_res_details[i] if i < len(struct_res_details) and isinstance(struct_res_details[i], dict) else {}

            # Extract continuous features, handling None/NaN/TypeErrors, default to np.nan
            def safe_float(value, default=np.nan):
                try:
                    f_val = float(value)
                    return f_val if np.isfinite(f_val) else default
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
        return []


if __name__ == "__main__":
    print("--- Starting Statistics Calculation using Dask ---")
    start_time = time.time()

    # Initialize Dask Client
    # Using LocalCluster provides more control and resources than the default threaded scheduler
    # Adjust n_workers and memory_limit based on your machine
    cluster = LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1, memory_limit='auto')
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}")

    full_dataset_uri = f"r2://{r2_bucket_name}/{R2_DATASET_DIR}"
    stats = {} # Dictionary to store results

    try:
        # Read Parquet into Dask DataFrame
        print(f"\nReading Dask DataFrame from: {full_dataset_uri}")
        ddf = dd.read_parquet(full_dataset_uri,
                              columns=COLUMNS_TO_READ,
                              storage_options=storage_options)
        
        print(f"Dask DataFrame created with {ddf.npartitions} partitions.")

        # --- Calculate Protein Stats ---
        print("\nCalculating protein-level statistics...")
        protein_start_time = time.time()
        # Apply the extraction function row-wise -> Series of numpy arrays
        # Need meta to specify the output type/shape for Dask
        protein_features_series = ddf.apply(extract_protein_features_numeric, axis=1,
                                            meta=('protein_features', 'object'))

        # Convert the Series of arrays into a Dask Array for efficient stats
        # WARNING: This step can consume significant memory on the Dask scheduler/workers
        # if the number of proteins is very large.
        print("  Stacking protein features into Dask Array (potentially memory intensive)...")
        protein_features_dask_array = da.stack(protein_features_series.values)
        print("  Dask Array for protein features created.")

        # Compute mean and std using Dask Array functions (handles NaNs)
        print("  Computing protein mean and std...")
        protein_mean = da.nanmean(protein_features_dask_array, axis=0)
        protein_std = da.nanstd(protein_features_dask_array, axis=0)

        # Trigger computation
        protein_mean_computed, protein_std_computed = da.compute(protein_mean, protein_std)
        protein_end_time = time.time()
        print(f"Protein stats computed in {protein_end_time - protein_start_time:.2f} seconds.")

        # --- Calculate Residue Stats ---
        print("\nCalculating residue-level statistics (may take significant time/memory)...")
        residue_start_time = time.time()

        # 1. Define function to apply to each partition to flatten residue features
        # This avoids creating a Dask Series containing large lists, improving memory efficiency.
        def partition_to_residue_features_df(df_partition):
            all_res_features = []
            for _, row in df_partition.iterrows():
                 # Extract list of feature lists for the protein
                 protein_residues = extract_residue_features_numeric(row)
                 # Extend the main list only if the extraction was successful
                 if protein_residues:
                     # Convert inner lists to numpy arrays for consistency if needed
                     # all_res_features.extend([np.array(res, dtype=np.float64) for res in protein_residues])
                     all_res_features.extend(protein_residues) # Keep as list of lists if helpers return lists

            # Create a DataFrame from the flattened list for this partition
            if not all_res_features:
                # Return empty DataFrame with correct columns if no residues in partition
                return pd.DataFrame(columns=[f'f_{i}' for i in range(EXPECTED_NUM_RESIDUE_FEATURES)], dtype=np.float64)

            # Specify column names for clarity
            col_names = [f'f_{i}' for i in range(EXPECTED_NUM_RESIDUE_FEATURES)]
            return pd.DataFrame(all_res_features, columns=col_names, dtype=np.float64)

        # 2. Apply the function using map_partitions
        # Define the expected output metadata (schema) for Dask
        meta_residue_df = pd.DataFrame(columns=[f'f_{i}' for i in range(EXPECTED_NUM_RESIDUE_FEATURES)], dtype=np.float64)
        print("  Applying residue extraction via map_partitions...")
        residue_df_exploded = ddf.map_partitions(partition_to_residue_features_df, meta=meta_residue_df)
        print("  Residue data exploded (intermediate Dask DataFrame created).")

        # 3. Compute stats on the exploded DataFrame's Dask Array representation
        # Ensure sufficient partitions or repartition if needed before array conversion for large data

        # Convert to Dask Array
        # WARNING: This step creates a potentially *very* large Dask Array containing
        # features for *all* residues across the dataset. Monitor memory usage closely.
        print("  Converting exploded residue DataFrame to Dask Array (potentially memory intensive)...")
        # lengths=True is important when converting from DataFrames where partition lengths are unknown
        residue_dask_array = residue_df_exploded.to_dask_array(lengths=True)
        print("  Dask Array for residue features created.")

        # Compute mean/std using Dask Array functions
        print("  Computing residue mean and std...")
        residue_mean = da.nanmean(residue_dask_array, axis=0)
        residue_std = da.nanstd(residue_dask_array, axis=0)

        # Trigger computation
        residue_mean_computed, residue_std_computed = da.compute(residue_mean, residue_std)
        residue_end_time = time.time()
        print(f"Residue stats computed in {residue_end_time - residue_start_time:.2f} seconds.")

        # --- Prepare and Save Stats ---
        # Add a small epsilon to std dev to prevent division by zero during normalization
        epsilon = 1e-7
        stats = {
            'protein_mean': protein_mean_computed.tolist(),
            'protein_std': np.maximum(protein_std_computed, 0.0, dtype=np.float64) + epsilon,
            'residue_mean': residue_mean_computed.tolist(),
            'residue_std': np.maximum(residue_std_computed, 0.0, dtype=np.float64) + epsilon
        }
        # Convert std devs back to lists for JSON serialization
        stats['protein_std'] = stats['protein_std'].tolist()
        stats['residue_std'] = stats['residue_std'].tolist()


        print(f"\nSaving statistics to: {STATS_OUTPUT_FILE}")
        try:
            # Ensure directory exists if STATS_OUTPUT_FILE includes a path
            output_dir = os.path.dirname(STATS_OUTPUT_FILE)
            if output_dir: # Only create if path is not just a filename
                os.makedirs(output_dir, exist_ok=True)

            with open(STATS_OUTPUT_FILE, 'w') as f:
                json.dump(stats, f, indent=4) # Use indent for readability
            print("Statistics saved successfully.")
            print("\n--- Statistics Summary ---")
            print(f"Protein Mean Vector Length: {len(stats['protein_mean'])}")
            print(f"Protein Std Dev Vector Length: {len(stats['protein_std'])}")
            print(f"Residue Mean Vector Length: {len(stats['residue_mean'])}")
            print(f"Residue Std Dev Vector Length: {len(stats['residue_std'])}")


        except IOError as e:
            print(f"ERROR: Could not write statistics file '{STATS_OUTPUT_FILE}': {e}")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during file writing: {e}")


    except ImportError as e:
        print(f"ERROR: Missing Dask library components: {e}. Try 'pip install dask[dataframe] distributed'")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during Dask statistics calculation: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed traceback during debugging
        sys.exit(1)
    finally:
        # Shutdown Dask client and cluster
        print("\nShutting down Dask client and cluster...")
        if 'client' in locals():
            client.close()
        if 'cluster' in locals():
            cluster.close()
        print("Dask resources released.")

    end_time = time.time()
    print(f"\n--- Statistics Calculation Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")