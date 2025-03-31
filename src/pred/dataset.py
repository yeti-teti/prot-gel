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
    def __init__(self, mean_std_path="mean_std.json", r2_config_env_path=".env"):
        """
            Initialize dataset reading from Cloudflare R2 Parquet using Dask DataFrame.
            Uses compute() on partitions for __getitem__, which is scalable in memory
            but potentially slow for random access.

            Args:
                mean_std_path (str): Path to the JSON file with normalization stats.
                r2_config_env_path (str): Path to the .env file with R2 credentials.
        """
        self.mean_std_path = mean_std_path
        self.r2_config_env_path = r2_config_env_path
        self.r2_dataset_dir = "integrated_data/viridiplantae_dataset_partitioned" # MUST match writer

        self._load_r2_credentials_and_connect()
        self.mean_std = self._load_mean_std()

        print("Initializing Dask DataFrame for ProteinDataset...")
        self.ddf = self._load_dask_dataframe()
        print(f"Dask DataFrame created with {self.ddf.npartitions} partitions.")

        # Calculate partition lengths and cumulative lengths for indexing
        print("Calculating partition divisions ... ")
        self.partition_lens = self.ddf.map_partitions(len).compute()
        self.cumulative_lens = np.cumsum([0] + list(self.partition_lens)) 
        self.data_length = self.cumulative_lens[-1] # Corrected
        print(f"Total dataset length: {self.data_length}")

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
        if not load_dotenv(dotenv_path=self.r2_config_env_path):
            print(f"Warning: .env file not found at {self.r2_config_env_path}.")

        access_key = os.getenv("CLOUDFARE_ACCESS_KEY")
        secret_key = os.getenv("CLOUDFARE_SECRET_KEY")
        account_id = os.getenv("CLOUDFARE_ACCOUNT_ID")
        self.r2_bucket_name = os.getenv("CLOUDFARE_BUCKET_NAME")
        endpoint = os.getenv("CLOUDFARE_ENDPOINT")

        if not endpoint and account_id: endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        if not all([access_key, secret_key, self.r2_bucket_name, endpoint]):
            print("ERROR: Missing Cloudflare R2 credentials/config in environment/.env"); sys.exit(1)

        # Store storage options for Dask
        self.storage_options = {'key': access_key, 'secret': secret_key, 'endpoint_url': endpoint}

    def _load_mean_std(self):
        print(f"Loading normalization statistics from: {self.mean_std_path}")
        try:
            with open(self.mean_std_path, 'r') as f: stats = json.load(f)
            # Convert back to numpy arrays
            for key in ['protein_mean', 'protein_std', 'residue_mean', 'residue_std']:
                stats[key] = np.array(stats[key])
            return stats
        except FileNotFoundError: print(f"ERROR: Stats file not found: {self.mean_std_path}. Run calculate_stats.py"); sys.exit(1)
        except Exception as e: print(f"ERROR: Loading stats file: {e}"); sys.exit(1)

    def _load_dask_dataframe(self):
        full_uri = f"r2://{self.r2_bucket_name}/{self.r2_dataset_dir}"
        try:
            # Specify columns needed for __getitem__ processing
            columns = ['uniprot_id', 'sequence', 'sequence_length', 'residue_features',
                       'structural_features', 'physicochemical_properties', 'domains', 'gelation']
            # Dask infers schema, reads partitioned parquet efficiently
            return dd.read_parquet(full_uri, columns=columns, storage_options=self.storage_options)
        except Exception as e:
            print(f"ERROR: Reading Parquet Dask DataFrame from {full_uri}: {e}")
            sys.exit(1)

    def __len__(self):
        return self.data_length
    
    def _get_partition(self, partition_index):
        """Gets partition using cache otherwise computes"""
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
            
            # Update cache (LRU Like)
            if len(self._partition_cache) >= self._cache_limit:
                try:
                    first_key = next(iter(self._partition_cache))
                    del self._partition_cache[first_key]
                except StopIteration:
                    pass
            
            self._partition_cache[partition_index] = partition_df
            return partition_df

    def __getitem__(self, idx):
        """Return processed features for a single protein."""
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
            # Sequence
            sequence = protein_data.get('sequence', '')
            if not sequence: raise ValueError(f"Empty sequence for protein index {idx}")
            seq_encoded = [self.aa_to_int.get(aa, self.unknown_aa_index) for aa in sequence]
            seq_tensor = torch.tensor(seq_encoded, dtype=torch.long)
            seq_len = len(sequence)

            # Residue Features
            res_features_list = []
            seq_res_feats = protein_data.get('residue_features', []) or []
            struct_list = protein_data.get('structural_features', []) or []
            struct = struct_list[0] if struct_list and isinstance(struct_list[0], dict) else {}
            struct_res_details = struct.get('residue_details', []) or []

            for i in range(seq_len):
                res_f = seq_res_feats[i] if i < len(seq_res_feats) and isinstance(seq_res_feats[i], dict) else {}
                res_d = struct_res_details[i] if i < len(struct_res_details) and isinstance(struct_res_details[i], dict) else {}

                acc_raw = res_d.get('relative_accessibility', 0.0)
                phi_raw = res_d.get('phi', 0.0)
                psi_raw = res_d.get('psi', 0.0)
                # Handle potential non-numeric types or NaN before normalization
                acc = float(acc_raw) if acc_raw is not None and isinstance(acc_raw, (int, float)) and np.isfinite(acc_raw) else 0.0
                phi = float(phi_raw) if phi_raw is not None and isinstance(phi_raw, (int, float)) and np.isfinite(phi_raw) else 0.0
                psi = float(psi_raw) if psi_raw is not None and isinstance(psi_raw, (int, float)) and np.isfinite(psi_raw) else 0.0

                cont = np.array([res_f.get('hydrophobicity', 0.0), res_f.get('polarity', 0.0),
                                 res_f.get('volume', 0.0), acc, phi, psi], dtype=np.float64) # Ensure float64
                # Check shapes before normalization
                if cont.shape != self.mean_std['residue_mean'].shape:
                    raise ValueError(f"Residue feature shape mismatch: Got {cont.shape}, Expected {self.mean_std['residue_mean'].shape}")

                norm_cont = (cont - self.mean_std['residue_mean']) / self.mean_std['residue_std']
                ss = res_d.get('secondary_structure', '-')
                ss_onehot = [1.0 if ss == cls else 0.0 for cls in self.ss_classes]
                res_features_list.append(np.concatenate((norm_cont, ss_onehot)))

            # Determine expected feature dim for empty tensor case
            num_residue_features = len(self.mean_std['residue_mean']) + len(self.ss_classes)
            residue_features_tensor = torch.tensor(np.array(res_features_list, dtype=np.float32), dtype=torch.float) if res_features_list else torch.empty((0, num_residue_features), dtype=torch.float)

            # Protein Features
            phy_prop = protein_data.get('physicochemical_properties', {}) or {}
            domains = protein_data.get('domains', []) or []
            prot_cont = [protein_data.get('sequence_length', 0), phy_prop.get('molecular_weight', 0.0),
                         phy_prop.get('aromaticity', 0.0), phy_prop.get('instability_index', 0.0),
                         phy_prop.get('isoelectric_point', 0.0), phy_prop.get('gravy', 0.0),
                         phy_prop.get('charge_at_pH_7', 0.0), struct.get('helix_percentage', 0.0),
                         struct.get('sheet_percentage', 0.0), struct.get('coil_percentage', 0.0)]
            domain_flags = [1.0 if any(d.get('accession') == gd for d in domains if isinstance(d, dict)) else 0.0
                           for gd in self.gelation_domains]
            prot_combined = np.array(prot_cont + domain_flags, dtype=np.float64) # Ensure float64

            if len(prot_combined) != len(self.mean_std['protein_mean']):
                 raise ValueError(f"Protein feature dimension mismatch! Expected {len(self.mean_std['protein_mean'])}, got {len(prot_combined)}")

            norm_prot = (prot_combined - self.mean_std['protein_mean']) / self.mean_std['protein_std']
            protein_features_tensor = torch.tensor(norm_prot, dtype=torch.float)

            # Label
            gelation_tensor = torch.tensor(1.0 if protein_data.get('gelation') == 'yes' else 0.0, dtype=torch.float)

            return {'sequence': seq_tensor, 'residue_features': residue_features_tensor,
                    'protein_features': protein_features_tensor, 'gelation': gelation_tensor,
                    'uniprot_id': protein_data.get('uniprot_id', 'N/A') }

        except Exception as e:
            print(f"ERROR: Feature processing failed for index {idx} (ID: {protein_data.get('uniprot_id', 'N/A')}): {e}")
            # Propagate error - dataloader collate_fn handles None return if desired
            # For now, re-raise to make errors obvious during development
            raise RuntimeError(f"Failed feature processing for item at index {idx}") from e
