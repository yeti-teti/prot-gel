import os
import json
import sys

import numpy as np
import pandas as pd

import dask.dataframe as dd
import dask.array as da

import pyarrow.fs as pafs
from dask.distributed import Client

from dotenv import load_dotenv

# --- Configuration ---
ENV_FILE_PATH = ".env"
R2_DATASET_DIR = "integrated_data/viridiplantae_dataset_partitioned" # MUST match writer
STATS_OUTPUT_FILE = "mean_std.json" # Output file for dataset.py
GELATION_DOMAINS = ["PF00190", "PF04702", "PF00234"]

# Columns needed from Parquet
COLUMNS_TO_READ = ['uniprot_id', 'sequence_length', 'physicochemical_properties',
                   'structural_features', 'residue_features', 'domains']

# --- Load Environment Variables & Configure R2 FS ---
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
    print("ERROR: Missing Cloudflare R2 credentials/config in environment/.env"); sys.exit(1)

# Prepare storage options for Dask
storage_options = {'key': r2_access_key, 'secret': r2_secret_key, 'endpoint_url': r2_endpoint}


# --- Helper Functions for Feature Extraction (to be used with Dask) ---
def extract_protein_features_numeric(row):
    """Extracts numeric protein features from a row (Pandas Series)."""
    try:
        phy_prop = row.get('physicochemical_properties', {}) or {}
        struct_list = row.get('structural_features', []) or []
        struct = struct_list[0] if struct_list and isinstance(struct_list[0], dict) else {}
        domains = row.get('domains', []) or []

        prot_cont = [row.get('sequence_length', 0), phy_prop.get('molecular_weight', 0.0),
                     phy_prop.get('aromaticity', 0.0), phy_prop.get('instability_index', 0.0),
                     phy_prop.get('isoelectric_point', 0.0), phy_prop.get('gravy', 0.0),
                     phy_prop.get('charge_at_pH_7', 0.0), struct.get('helix_percentage', 0.0),
                     struct.get('sheet_percentage', 0.0), struct.get('coil_percentage', 0.0)]
        domain_flags = [1.0 if any(d.get('accession') == gd for d in domains if isinstance(d, dict)) else 0.0
                       for gd in GELATION_DOMAINS]
        # Ensure float type for consistency
        return np.array(prot_cont + domain_flags, dtype=np.float64)
    except Exception:
        # Return NaNs or expected shape with NaNs on error to avoid breaking aggregation
        # Number of features = 10 continuous + len(GELATION_DOMAINS)
        return np.full(10 + len(GELATION_DOMAINS), np.nan, dtype=np.float64)

def extract_residue_features_numeric(row):
    """Extracts numeric residue features as a list of lists for a single protein row."""
    residue_data = []
    try:
        seq_len = row.get('sequence_length', 0)
        if seq_len == 0: return [] # Handle empty sequence case

        seq_res_feats = row.get('residue_features', []) or []
        struct_list = row.get('structural_features', []) or []
        struct = struct_list[0] if struct_list and isinstance(struct_list[0], dict) else {}
        struct_res_details = struct.get('residue_details', []) or []

        for i in range(seq_len):
            res_f = seq_res_feats[i] if i < len(seq_res_feats) and isinstance(seq_res_feats[i], dict) else {}
            res_d = struct_res_details[i] if i < len(struct_res_details) and isinstance(struct_res_details[i], dict) else {}

            acc_raw = res_d.get('relative_accessibility', 0.0)
            phi_raw = res_d.get('phi', 0.0)
            psi_raw = res_d.get('psi', 0.0)

            # Handle None/NaN before conversion
            acc = float(acc_raw) if acc_raw is not None and np.isfinite(acc_raw) else np.nan
            phi = float(phi_raw) if phi_raw is not None and np.isfinite(phi_raw) else np.nan
            psi = float(psi_raw) if psi_raw is not None and np.isfinite(psi_raw) else np.nan

            residue_data.append([
                res_f.get('hydrophobicity', np.nan), res_f.get('polarity', np.nan),
                res_f.get('volume', np.nan), acc, phi, psi
            ])
        # Return list of lists/arrays, inner is [hydro, polar, vol, acc, phi, psi]
        return residue_data # Each element is a list of features for one residue
    except Exception:
        # Return empty list on error for this row
        return []

if __name__ == "__main__":
    # Dask Client for dashboard and resource management
    # client = Client() # Uses local scheduler by default
    # print(f"Dask dashboard link: {client.dashboard_link}")

    print("--- Starting Statistics Calculation using Dask ---")
    full_dataset_uri = f"r2://{r2_bucket_name}/{R2_DATASET_DIR}"

    try:
        # Read Parquet into Dask DataFrame
        print(f"Reading Dask DataFrame from: {full_dataset_uri}")
        ddf = dd.read_parquet(full_dataset_uri, columns=COLUMNS_TO_READ, storage_options=storage_options)
        print(f"Dask DataFrame created with {ddf.npartitions} partitions.")
        # Persist ddf in memory (Requires huge resources)
        # ddf = ddf.persist()

        # --- Calculate Protein Stats ---
        print("Calculating protein-level statistics...")
        # Apply the extraction function row-wise -> Series of numpy arrays
        # Need meta to specify the output type/shape for Dask
        protein_features_series = ddf.apply(extract_protein_features_numeric, axis=1, meta=('protein_features', 'object'))

        # Convert the Series of arrays into a Dask Array for efficient stats
        # This can be memory intensive if arrays are large/numerous
        protein_features_dask_array = da.stack(protein_features_series.values)

        # Compute mean and std using Dask Array functions (handles NaNs)
        protein_mean = da.nanmean(protein_features_dask_array, axis=0).compute()
        protein_std = da.nanstd(protein_features_dask_array, axis=0).compute()
        print("Protein stats computed.")

        # --- Calculate Residue Stats ---
        print("Calculating residue-level statistics (may take time)...")
        # 1. Apply function to get list of residue feature lists per row
        # residue_features_per_protein = ddf.apply(extract_residue_features_numeric, axis=1, meta=('residue_list', 'object'))

        # 2. Explode the lists using map_partitions for better memory management
        def partition_to_residue_features(df_partition):
            all_res_features = []
            for _, row in df_partition.iterrows():
                 protein_residues = extract_residue_features_numeric(row)
                 if protein_residues:
                      all_res_features.extend([np.array(res, dtype=np.float64) for res in protein_residues])
            if not all_res_features:
                 return pd.DataFrame(columns=['hydro', 'polar', 'vol', 'acc', 'phi', 'psi'], dtype=np.float64)
            return pd.DataFrame(all_res_features, columns=['hydro', 'polar', 'vol', 'acc', 'phi', 'psi'])

        meta_residue_df = pd.DataFrame(columns=['hydro', 'polar', 'vol', 'acc', 'phi', 'psi'], dtype=np.float64)
        residue_df_exploded = ddf.map_partitions(partition_to_residue_features, meta=meta_residue_df)
        print("  Residue data exploded (intermediate Dask DataFrame created).")

        # 3. Compute stats on the exploded DataFrame's Dask Array representation
        # Ensure sufficient partitions or repartition if needed before array conversion for large data
        # residue_df_exploded = residue_df_exploded.repartition(npartitions=...)
        residue_dask_array = residue_df_exploded.to_dask_array(lengths=True)
        print("  Calculating mean/std on residue Dask Array...")
        residue_mean = da.nanmean(residue_dask_array, axis=0).compute()
        residue_std = da.nanstd(residue_dask_array, axis=0).compute()
        print("Residue stats computed.")


        # --- Prepare and Save Stats ---
        epsilon = 1e-6
        stats = {
            'protein_mean': protein_mean.tolist(),
            'protein_std': np.maximum(protein_std, epsilon).tolist(),
            'residue_mean': residue_mean.tolist(),
            'residue_std': np.maximum(residue_std, epsilon).tolist()
        }

        print(f"Saving statistics to: {STATS_OUTPUT_FILE}")
        try:
            # Ensure directory exists if STATS_OUTPUT_FILE includes a path
            # os.makedirs(os.path.dirname(STATS_OUTPUT_FILE), exist_ok=True)
            with open(STATS_OUTPUT_FILE, 'w') as f: json.dump(stats, f, indent=2)
            print("Statistics saved successfully.")
        except Exception as e: print(f"ERROR: Could not write statistics file {STATS_OUTPUT_FILE}: {e}")

    except ImportError as e:
        print(f"ERROR: Missing Dask library components: {e}. Try 'pip install dask[dataframe] distributed'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during Dask statistics calculation: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed traceback
        sys.exit(1)
    # finally:
        # Shutdown Dask client
        # if 'client' in locals(): client.close()

    print("--- Statistics Calculation Finished ---")