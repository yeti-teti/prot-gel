import os
import json
import sys
import bisect

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import dask.dataframe as dd
import pyarrow.fs as pafs

from dotenv import load_dotenv


class ProteinDataset(Dataset):
    def __init__(self, r2_dataset_path, mean_std_path="mean_std.json", r2_config_env_path=".env", r2_bucket_name=None):
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

        self._load_r2_credentials_and_connect()
        self.mean_std = self._load_mean_std()

        print("Initializing Dask DataFrame for ProteinDataset...")
        self.ddf = self._load_dask_dataframe()
        print(f"Dask DataFrame created with {self.ddf.npartitions} partitions for path: {r2_dataset_path}")

        # Calculate partition lengths and cumulative lengths for indexing
        print(f"Calculating partition divisions for {r2_dataset_path} ... ")
        self.partition_lens = self.ddf.map_partitions(len).compute()
        self.cumulative_lens = np.cumsum([0] + self.partition_lens)
        self.data_length = self.cumulative_lens[-1] # Total length is the last cumulative value
        print(f"Total dataset length for {r2_dataset_path}: {self.data_length}")
        if self.data_length == 0:
            print(f"Warning: Dataset at {r2_dataset_path} is empty")

        # Cache for computed partitions
        self._partition_cache = {}
        self._cache_limit = 2

        # Feature constants
        self.ss_classes = ['H', 'G', 'I', 'E', 'B', 'T', 'S', '-']
        self.aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_int = {aa: i + 1 for i, aa in enumerate(self.aa_list)}
        self.unknown_aa_index = len(self.aa_list) + 1 # 21
        self.padding_idx = 0
        self.gelation_domains = ["PF00190", "PF04702", "PF00234"]

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

        # Store storage options for Dask
        self.storage_options = {'key': access_key, 'secret': secret_key, 'endpoint_url': endpoint}
        print(f"Dataset configured for R2 Endpoint: {endpoint}, Bucket: {self.r2_bucket_name}")

    def _load_mean_std(self):
        """Loads normalization statistics from the specified JSON file."""
        print(f"Loading normalization statistics from: {self.mean_std_path}")
        try:
            with open(self.mean_std_path, 'r') as f:
                stats = json.load(f)
            # Convert back to numpy arrays
            for key in ['protein_mean', 'protein_std', 'residue_mean', 'residue_std']:
                stats[key] = np.array(stats[key])
            return stats
        except FileNotFoundError: 
            print(f"ERROR: Stats file not found: {self.mean_std_path}. Run calculate_stats.py"); sys.exit(1)
        except Exception as e: 
            print(f"ERROR: Loading stats file: {e}"); sys.exit(1)

    def _load_dask_dataframe(self):
        """Loads the Dask DataFrame from the specified R2 Parquet path."""

        # Full URI for Dask
        full_uri = f"r2://{self.r2_bucket_name}/{self.r2_dataset_dir}"
        print(f"Reading full Parquet dataset form: {full_uri}")
        try:
            # Specify columns needed for __getitem__ processing
            columns = ['uniprot_id', 'sequence', 'sequence_length', 'residue_features',
                       'structural_features', 'physicochemical_properties', 'domains', 'gelation']
            # Dask infers schema and reads the partitioned Parquet dataset efficiently
            ddf = dd.read_parquet(full_uri, columns=columns, storage_options=self.storage_options)
            print(f"Successfully initialized Dask DataFrame from {full_uri}")
            return ddf
        except Exception as e:
            print(f"ERROR: Reading Parquet Dask DataFrame from {full_uri}: {e}")
            sys.exit(1)

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return self.data_length
    
    def _get_partition(self, partition_index):
        """Gets a specific partition DataFrame, using cache or computing it."""
        if partition_index in self._partition_cache:
            return self._partition_cache[partition_index]
        else:
            # Compute the required parition into a Pandas DataFrame
            print(f"  Computing partition {partition_index}...")
            try:
                partition_df = self.ddf.get_partition(partition_index).compute()
                print(f"  Partition {partition_index} computed (size: {len(partition_df)}).")
            except Exception as e:
                print(f"ERROR: Failed to compute partition {partition_index}: {e}")
                raise
            
            # Update cache (LRU Like removing oldest)
            if len(self._partition_cache) >= self._cache_limit:
                try:
                    first_key = next(iter(self._partition_cache))
                    del self._partition_cache[first_key]
                except StopIteration:
                    pass # Cache already empty
            
            self._partition_cache[partition_index] = partition_df
            return partition_df

    def __getitem__(self, idx):
        """Gets and processes features for a single protein at the given index."""
        if not 0 <= idx < self.data_length:
            raise IndexError(f"Index {idx} out of bounds for dataset length {self.data_length}")

        # Finding which parition the index belongs to 
        # bisect_right finds the insertion point, subtract 1 for the correct partition index
        partition_index = bisect.bisect_right(self.cummulative_lens, idx) - 1
        # Calcualte index within that partition
        local_idx = idx - self.cumulative_lens[partition_index]

        try:
            # Get the partition (computed if not cached)
            partition_df = self._get_partition(partition_index)
            if local_idx >= len(partition_df):
                raise IndexError(f"Calculated local index {local_idx} is out of bounds for computed partition {partition_index} (size {len(partition_df)})")
            protein_data = partition_df.iloc[local_idx]
        except Exception as e:
            print(f"ERROR: Failed to get item at global index {idx} (partition {partition_index}, local index {local_idx}): {e}")
            raise RuntimeError(f"Failed to retrieve item at index {idx}") from e
        

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
            seq_res_feats = protein_data.get('residue_features', []) or []
            struct_list = protein_data.get('structural_features', []) or []
            struct = struct_list[0] if struct_list and isinstance(struct_list[0], dict) else {}
            struct_res_details = struct.get('residue_details', []) or []

            expected_res_len_stats = len(self.mean_std['residue_mean']) # Should be 6
            expected_prot_len_stats = len(self.mean_std['protein_mean']) # Should be 13

            # Iterate up to the actual sequence length
            for i in range(seq_len):
                # Get features for residue i, handling potential index errors or type issues
                res_f = seq_res_feats[i] if i < len(seq_res_feats) and isinstance(seq_res_feats[i], dict) else {}
                res_d = struct_res_details[i] if i < len(struct_res_details) and isinstance(struct_res_details[i], dict) else {}

                # Extract continuous features, handling None/NaNs, default to 0.0 if missing
                # AAIndex features
                hydrophobicity = float(res_f.get('hydrophobicity', 0.0)) if np.isfinite(res_f.get('hydrophobicity', 0.0)) else 0.0
                polarity = float(res_f.get('polarity', 0.0)) if np.isfinite(res_f.get('polarity', 0.0)) else 0.0
                volume = float(res_f.get('volume', 0.0)) if np.isfinite(res_f.get('volume', 0.0)) else 0.0
                # DSSP features
                acc_raw = res_d.get('relative_accessibility')
                acc = float(acc_raw) if acc_raw is not None and isinstance(acc_raw, (int, float)) and np.isfinite(acc_raw) else 0.0
                phi_raw = res_d.get('phi')
                phi = float(phi_raw) if phi_raw is not None and isinstance(phi_raw, (int, float)) and np.isfinite(phi_raw) else 0.0
                psi_raw = res_d.get('psi')
                psi = float(psi_raw) if psi_raw is not None and isinstance(psi_raw, (int, float)) and np.isfinite(psi_raw) else 0.0

                cont_features = np.array([hydrophobicity, polarity, volume, acc, phi, psi], dtype=np.float64)

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
                residue_features_tensor = torch.empty((0, num_residue_feat_dim), dtype=torch.float)

            # Check consistency between sequence length and residue features dimension
            if seq_len != residue_features_tensor.shape[0]:
                print(f"WARNING: Sequence length ({seq_len}) mismatch with processed residue features ({residue_features_tensor.shape[0]}) for ID {uniprot_id}. Padding/truncating residue features tensor.")
                # Adjust residue features tensor to match sequence length (e.g., pad with zeros or truncate)
                if residue_features_tensor.shape[0] > seq_len:
                    residue_features_tensor = residue_features_tensor[:seq_len, :]
                else:
                    padding_needed = seq_len - residue_features_tensor.shape[0]
                    padding_tensor = torch.zeros((padding_needed, num_residue_feat_dim), dtype=torch.float)
                    residue_features_tensor = torch.cat((residue_features_tensor, padding_tensor), dim=0)


            # Protein Features Processing
            phy_prop_raw = protein_data.get('physicochemical_properties', {})
            phy_prop = phy_prop_raw if isinstance(phy_prop_raw, dict) else {}
            domains_raw = protein_data.get('domains', [])
            domains = domains_raw if isinstance(domains_raw, list) else []

            # Extract continuous protein features, handle missing with 0.0
            prot_cont = [
                float(protein_data.get('sequence_length', 0.0)), # Already have seq_len, but use stored value for consistency
                float(phy_prop.get('molecular_weight', 0.0)),
                float(phy_prop.get('aromaticity', 0.0)),
                float(phy_prop.get('instability_index', 0.0)),
                float(phy_prop.get('isoelectric_point', 0.0)),
                float(phy_prop.get('gravy', 0.0)),
                float(phy_prop.get('charge_at_pH_7', 0.0)),
                float(struct.get('helix_percentage', 0.0)),
                float(struct.get('sheet_percentage', 0.0)),
                float(struct.get('coil_percentage', 0.0))
            ]
            # Ensure all are finite, replace NaN/inf with 0.0
            prot_cont = [x if np.isfinite(x) else 0.0 for x in prot_cont]

            # Domain flags (binary categorical)
            domain_flags = [1.0 if any(isinstance(d, dict) and d.get('accession') == gd for d in domains) else 0.0
                            for gd in self.gelation_domains]

            prot_combined = np.array(prot_cont + domain_flags, dtype=np.float64)

            # Validate shape before normalization
            if prot_combined.shape[0] != expected_prot_len_stats:
                raise ValueError(f"Protein feature shape mismatch for ID {uniprot_id}: Got {prot_combined.shape[0]}, Expected {expected_prot_len_stats}")

            # Normalize protein features
            norm_prot = (prot_combined - self.mean_std['protein_mean']) / self.mean_std['protein_std']
            protein_features_tensor = torch.tensor(norm_prot, dtype=torch.float)

            # Label Processing
            gelation_label = protein_data.get('gelation', 'no') # Default to 'no' if missing
            gelation_tensor = torch.tensor(1.0 if gelation_label == 'yes' else 0.0, dtype=torch.float)

            return {
                'sequence': seq_tensor,
                'residue_features': residue_features_tensor,
                'protein_features': protein_features_tensor,
                'gelation': gelation_tensor,
                'uniprot_id': uniprot_id # Include the ID here
            }

        except Exception as e:
            # Log error with more context
            print(f"ERROR: Feature processing failed for index {idx} (ID: {protein_data.get('uniprot_id', 'N/A')}): {e}")
            # Re-raise to make errors obvious during development/debugging
            # In production, returning None might be preferred if collate_fn handles it robustly
            raise RuntimeError(f"Failed feature processing for item at index {idx}") from e