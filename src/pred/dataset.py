import os
import json
import sys
import bisect
import gc
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from dotenv import load_dotenv

# Constants mirroring calculate_stats.py (Important for validation)
GELATION_DOMAINS = ["PF00190", "PF04702", "PF00234"]
# Expecting 10 continuous + num_gelation_domains binary flags
EXPECTED_NUM_PROTEIN_FEATURES = 10 + len(GELATION_DOMAINS)
# Expecting 6 continuous residue features (hydro, polar, vol, acc, phi, psi)
EXPECTED_NUM_RESIDUE_FEATURES_CONTINUOUS = 6

class ProteinDataset(Dataset):
    def __init__(self, r2_dataset_path, mean_std_path="mean_std.json", r2_config_env_path=".env", r2_bucket_name=None, cache_limit=2):
        """
            Args:
                r2_dataset_path (str): Path WITHIN the R2 bucket to the partitioned Parquet dataset directory
                                        (e.g., "integrated_data/train_split_parquet").
                mean_std_path (str): Path to the JSON file with normalization stats.
                r2_config_env_path (str): Path to the .env file with R2 credentials.
                r2_bucket_name (str, optional): R2 bucket name. If None, tries to load from .env.
         """
        self.r2_dataset_path = r2_dataset_path
        self.mean_std_path = mean_std_path
        self.r2_config_env_path = r2_config_env_path
        self.r2_bucket_name_arg = r2_bucket_name
        self.cache_limit = cache_limit

        # Columns required for processing in __getitem__
        self.columns_to_read = [
            'uniprot_id', 'sequence', 'sequence_length',
            'physicochemical_properties', 'residue_features',
            'structural_features', 'domains', 'gelation'
        ]

        self.r2_fs = None
        self._load_r2_credentials_and_connect()

        self.mean_std = self._load_mean_std()

        print("Scanning Parquet dataset metadata...")
        self.parquet_files = []
        self.file_row_counts = []
        self.cumulative_row_counts = []
        self.data_length = 0
        self._scan_parquet_dataset()
        print(f"Dataset initialized. Found {len(self.parquet_files)} Parquet files.")
        print(f"Total dataset length: {self.data_length}")

        self._dataframe_cache = OrderedDict()

        # Feature constants
        self.ss_classes = ['H', 'G', 'I', 'E', 'B', 'T', 'S', '-']
        self.aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_int = {aa: i + 1 for i, aa in enumerate(self.aa_list)}
        self.unknown_aa_index = len(self.aa_list) + 1 # 21
        self.padding_idx = 0
        self.gelation_domains = GELATION_DOMAINS

    def _load_r2_credentials_and_connect(self):
        """Loads R2 credentials from .env and sets bucket name and storage options."""

        print(f"Loading R2 credentials from: {self.r2_config_env_path}")
        if not load_dotenv(dotenv_path=self.r2_config_env_path):
            print(f"Warning: .env file not found at {self.r2_config_env_path}.")

        access_key = os.getenv("CLOUDFARE_ACCESS_KEY")
        secret_key = os.getenv("CLOUDFARE_SECRET_KEY")
        account_id = os.getenv("CLOUDFARE_ACCOUNT_ID")
        self.r2_bucket_name = self.r2_bucket_name_arg or os.getenv("CLOUDFARE_BUCKET_NAME")
        endpoint = os.getenv("CLOUDFARE_ENDPOINT")

        if not endpoint and account_id: 
            endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        if not all([access_key, secret_key, self.r2_bucket_name, endpoint]):
            print("ERROR: Missing Cloudflare R2 credentials/config in environment/.env"); sys.exit(1)

        try:
            self.r2_fs = pafs.S3FileSystem(
                endpoint_override = endpoint,
                access_key = access_key,
                secret_key = secret_key,
                scheme = 'https'
            )
            # Test connection by getting bucket info
            print(f"Testing R2 connection to bucket '{self.r2_bucket_name}'...")
            self.r2_fs.get_file_info(f"{self.r2_bucket_name}/") # Check connectivity
            print("R2 Filesystem connection successful.")
            print(f"Dataset configured for R2 Endpoint: {endpoint}, Bucket: {self.r2_bucket_name}")
        except Exception as e:
            print(f"ERROR: Failed to configure or connect R2 filesystem for bucket '{self.r2_bucket_name}': {e}")
            sys.exit(1)

    def _load_mean_std(self):
        """Loads normalization statistics from the specified JSON file."""
        print(f"Loading normalization statistics from: {self.mean_std_path}")
        try:
            with open(self.mean_std_path, 'r') as f:
                stats = json.load(f)
            # Convert back to numpy arrays
            for key in ['protein_mean', 'protein_std', 'residue_mean', 'residue_std']:
                stats[key] = np.array(stats[key], dtype=np.float32)
            print("Normalization stats loaded.")
            return stats
        except FileNotFoundError: 
            print(f"ERROR: Stats file not found: {self.mean_std_path}. Run calculate_stats.py"); sys.exit(1)
        except Exception as e: 
            print(f"ERROR: Loading stats file: {e}"); sys.exit(1)

    def _scan_parquet_dataset(self):
        """Scans the R2 directory for Parquet files and reads metadata to get row counts"""
        full_dataset_uri = f"{self.r2_bucket_name}/{self.r2_dataset_path}"
        print(f"Scanning Parquet files in: s3:://{full_dataset_uri}")
        try:
            selector = pa.fs.FileSelector(full_dataset_uri, recursive=True)
            file_infos = self.r2_fs.get_file_info(selector)

            found_files = []
            for file_info in file_infos:
                if file_info.is_file and file_info.path.endswith('.parquet'):
                    found_files.append(file_info.path)
            if not found_files:
                print(f"Warning: No .parquet files found in s3://{full_dataset_uri}")
                self.data_length = 0
                self.cumulative_row_counts = np.array([0])
                return

            self.parquet_files = sorted(found_files)
            print(f"Found {len(self.parquet_files)} Parquet files. Reading row counts...")
            current_total_rows = 0
            row_counts = []
            cumulative_rows = [0]

            for i, file_path in enumerate(self.parquet_files):
                try:
                    metadata = pq.read_metadata(file_path, filesystem=self.r2_fs)
                    num_rows = metadata.num_rows
                    row_counts.append(num_rows)
                    current_total_rows += num_rows
                    cumulative_rows.append(current_total_rows)
                except Exception as e:
                    print(f"\nError: Failed to read metadata for file: {file_path}. Error {e}")
                    row_counts.append(0)
                    cumulative_rows.append(current_total_rows)
            print(f"\n Finished reading metadata. Total rows : {current_total_rows}")
            self.file_row_counts = np.array(row_counts)
            self.cumulative_row_counts = np.array(cumulative_rows)
            self.data_length = current_total_rows
        except Exception as e:
            print(f"Error failed to read data from parquet {e}")
            self.data_length = 0
            self.parquet_files = []
            self.file_row_counts = np.array([])
            self.cumulative_row_counts = np.array([0])

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return self.data_length
    
    def _get_dataframe_for_index(self, idx):
        """
            Finds the correct Parquet file for global index, loads it into a Pandas Dataframe (using cache), and returns the DF and local index within that DF.
        """
        if not 0 <= idx < self.data_length:
            raise IndexError(f"Global index {idx} out of bounds. Data length {self.data_length}")
        
        file_index = bisect.bisect_right(self.cumulative_row_counts, idx) - 1

        if not (0 <= file_index < len(self.parquet_files)):
            raise RuntimeError(f"Could not determine valid file index for global index {idx}. Cumulative counts: {self.cumulative_row_counts}")
        
        # Calculate the index within the specific file's DataFrame
        local_idx = idx - self.cumulative_row_counts[file_index]

        # Cache Handling
        if file_index in self._dataframe_cache:
            self._dataframe_cache.move_to_end(file_index)
            return self._dataframe_cache[file_index], local_idx
        else:
            file_path = self.parquet_files[file_index]
            print(f"  Cache miss. Loading file index {file_index}: s3://{file_path}...")
            try:
                table = pq.read_table(
                    file_path,
                    filesystem = self.r2_fs,
                    columns = self.columns_to_read
                )
                partition_df = table.to_pandas()
                del table
                gc.collect()
                print(f"  Loaded DataFrame for file index {file_index} (Shape: {partition_df.shape}).")

                if len(self._dataframe_cache) >= self.cache_limit:
                    oldest_key, _ = self._dataframe_cache.popitem(last=False)
                    print(f"  Cache limit reached. Removed file index {oldest_key} from cache.")
                self._dataframe_cache[file_index] = partition_df

                return partition_df, local_idx
            except Exception as e:
                raise RuntimeError(f"Failed to load data for index {idx} from file {file_path}") from e

    def __getitem__(self, idx):
        """Gets and processes features for a single protein at the given global index."""
        if self.data_length == 0:
             raise IndexError("Cannot get item from an empty dataset.")

        try:
            # Get the Pandas DataFrame and local index for the requested global index
            partition_df, local_idx = self._get_dataframe_for_index(idx)

            # Access the specific row using the local index
            if local_idx >= len(partition_df):
                raise IndexError(f"Calculated local index {local_idx} is out of bounds for DataFrame from file index {bisect.bisect_right(self.cumulative_row_counts, idx) - 1} (size {len(partition_df)})")
            protein_data = partition_df.iloc[local_idx] # protein_data is now a Pandas Series
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve or access item at global index {idx}") from e

        # --- Feature Processing (Pandas Series 'protein_data') ---
        try:
            uniprot_id = protein_data.get('uniprot_id', f"Missing ID: {idx}")

            # Sequence Encoding
            sequence = protein_data.get('sequence', '')
            if not sequence or not isinstance(sequence, str): 
                # Handle missing/invalid sequence
                print(f"WARNING: Empty or invalid sequence for index {idx} (ID: {uniprot_id}). Returning empty tensors.")
                # Return empty/zero tensors matching expected output structure
                num_residue_feat_dim = len(self.mean_std['residue_mean']) + len(self.ss_classes)
                num_protein_feat_dim = len(self.mean_std['protein_mean'])
                return {
                    'sequence': torch.zeros(0, dtype=torch.long),
                    'residue_features': torch.zeros((0, num_residue_feat_dim), dtype=torch.float),
                    'protein_features': torch.zeros(num_protein_feat_dim, dtype=torch.float),
                    'gelation': torch.tensor(0.0, dtype=torch.float), # Default label? Or skip?
                    'uniprot_id': uniprot_id
                }

            seq_encoded = [self.aa_to_int.get(aa, self.unknown_aa_index) for aa in sequence]
            seq_tensor = torch.tensor(seq_encoded, dtype=torch.long)
            seq_len = len(sequence)

            # Residue Features Processing
            res_features_list = []
            seq_res_feats = protein_data.get('residue_features', [])
            struct_list = protein_data.get('structural_features', [])
            struct = struct_list[0] if struct_list and isinstance(struct_list[0], dict) else {}
            struct_res_details = struct.get('dssp_residue_details', [])

            expected_res_len_stats = len(self.mean_std['residue_mean']) # Should be 6
            expected_prot_len_stats = len(self.mean_std['protein_mean']) # Should be 13

            # Iterate up to the actual sequence length
            for i in range(seq_len):
                # Get features for residue i, handling potential index errors or type issues
                res_f = seq_res_feats[i] if i < len(seq_res_feats) and isinstance(seq_res_feats[i], dict) else {}
                res_d = struct_res_details[i] if i < len(struct_res_details) and isinstance(struct_res_details[i], dict) else {}

                def safe_float(value, default=0.0): # Default to 0.0 for features if missing/invalid
                    try:
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            return float(value)
                        if value is not None:
                            f_val = float(value)
                            return f_val if np.isfinite(f_val) else default
                        return default
                    except (TypeError, ValueError):
                        return default

                # Extract continuous features, handling None/NaNs, default to 0.0 if missing
                # Match the order expected by calculate_stats.py for residue_mean/std
                cont_features = np.array([
                    # AAIndex features
                    safe_float(res_f.get('hydrophobicity')),
                    safe_float(res_f.get('polarity')),
                    safe_float(res_f.get('volume')),
                    # DSSP features
                    safe_float(res_d.get('relative_accessibility')),
                    safe_float(res_d.get('phi')),
                    safe_float(res_d.get('psi'))
                ], dtype=np.float64)

                # Validate shape before normalization
                if cont_features.shape[0] != expected_res_len_stats:
                    raise ValueError(f"Residue feature shape mismatch for ID {uniprot_id} at residue {i}: Got {cont_features.shape[0]}, Expected {expected_res_len_stats}")

                # Normalize continuous features
                norm_cont = (cont_features - self.mean_std['residue_mean']) / self.mean_std['residue_std']

                # Secondary structure one-hot encoding
                ss = res_d.get('secondary_structure', '-') # Default to '-' if missing
                ss_onehot = [1.0 if ss == cls else 0.0 for cls in self.ss_classes]

                # Concatenate normalized continuous and categorical features
                res_features_list.append(np.concatenate((norm_cont, ss_onehot)))

            # Convert list of features to a tensor
            num_residue_feat_dim = expected_res_len_stats + len(self.ss_classes) # 6 + 8 = 14
            if res_features_list:
                residue_features_tensor = torch.tensor(np.array(res_features_list, dtype=np.float32), dtype=torch.float)
            else:
                # Handle case where sequence might exist but no features could be processed (should be rare if seq check passed)
                residue_features_tensor = torch.empty((0, num_residue_feat_dim), dtype=torch.float32)

            # Check consistency between sequence length and residue features dimension
            if seq_len != residue_features_tensor.shape[0]:
                print(f"WARNING: Sequence length ({seq_len}) mismatch with processed residue features ({residue_features_tensor.shape[0]}) for ID {uniprot_id}. Padding/truncating residue features tensor.")
                # Adjust residue features tensor to match sequence length (e.g., pad with zeros or truncate)
                if residue_features_tensor.shape[0] > seq_len:
                    residue_features_tensor = residue_features_tensor[:seq_len, :]
                else:
                    padding_needed = seq_len - residue_features_tensor.shape[0]
                    padding_tensor = torch.zeros((padding_needed, num_residue_feat_dim), dtype=torch.float32)
                    residue_features_tensor = torch.cat((residue_features_tensor, padding_tensor), dim=0)


            # Protein Features Processing
            phy_prop_raw = protein_data.get('physicochemical_properties', {})
            phy_prop = phy_prop_raw if isinstance(phy_prop_raw, dict) else {}
            domains_raw = protein_data.get('domains', [])
            domains = domains_raw if isinstance(domains_raw, list) else []

            # Extract continuous protein features, handle missing with 0.0
            prot_cont = [
                safe_float(protein_data.get('sequence_length', 0.0)),
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

            # Domain flags (binary categorical)
            domain_flags = [1.0 if any(isinstance(d, dict) and d.get('accession') == gd for d in domains) else 0.0
                            for gd in self.gelation_domains]

            prot_combined = np.array(prot_cont + domain_flags, dtype=np.float64)

            # Validate shape before normalization
            if prot_combined.shape[0] != expected_prot_len_stats:
                raise ValueError(f"Protein feature shape mismatch for ID {uniprot_id}: Got {prot_combined.shape[0]}, Expected {expected_prot_len_stats}")

            # Normalize protein features
            norm_prot = (prot_combined - self.mean_std['protein_mean']) / self.mean_std['protein_std']
            protein_features_tensor = torch.tensor(norm_prot, dtype=torch.float32)

            # Label Processing
            gelation_label = protein_data.get('gelation', 'no') # Default to 'no' if missing
            gelation_tensor = torch.tensor(1.0 if gelation_label == 'yes' else 0.0, dtype=torch.float32)

            return {
                'sequence': seq_tensor,
                'residue_features': residue_features_tensor,
                'protein_features': protein_features_tensor,
                'gelation': gelation_tensor,
                'uniprot_id': uniprot_id # Include the ID here
            }

        except Exception as e:
            print(f"ERROR: Feature processing failed for index {idx} (ID: {protein_data.get('uniprot_id', 'N/A')}): {e}")
            raise RuntimeError(f"Failed feature processing for item at index {idx}") from e