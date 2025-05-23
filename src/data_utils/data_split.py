import argparse
import gc
import os
import sys
import time
import json

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from dotenv import load_dotenv

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.model_selection import train_test_split


#  Configuration & R2 Setup 
ENV_FILE_PATH = ".env"
# R2 Path for the INPUT dataset 
DEFAULT_R2_INPUT_DIR = "integrated_data/viridiplantae_dataset_partitioned_from_json"
# Default R2 Paths for the OUTPUT train/test Parquet datasets
DEFAULT_R2_TRAIN_DIR = "integrated_data/train_split_parquet"
DEFAULT_R2_TEST_DIR = "integrated_data/test_split_parquet"

# Pyarrow schema for output
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

def setup_r2_fs(env_path=ENV_FILE_PATH):
    """
        Loads R2 credentials and returns bucket_name, storage_options for Dask
    """
    print(f"Loading R2 credentials from: {env_path}")
    if not load_dotenv(dotenv_path=env_path):
        print(f"Warning: .env file not found at {env_path}")

    access_key = os.getenv("CLOUDFARE_ACCESS_KEY")
    secret_key = os.getenv("CLOUDFARE_SECRET_KEY")
    account_id = os.getenv("CLOUDFARE_ACCOUNT_ID")
    bucket_name = os.getenv("CLOUDFARE_BUCKET_NAME")
    endpoint = os.getenv("CLOUDFARE_ENDPOINT")

    if not endpoint and account_id:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

    if not all([access_key, secret_key, bucket_name, endpoint]):
        print("ERROR: Missing Cloudflare R2 credentials/config in environment/.env")
        sys.exit(1)

    print(f"Using R2 Endpoint: {endpoint}, Bucket: {bucket_name}")
    
    try:
        print("Testing R2 connection using s3fs")
        fs = pafs.S3FileSystem(
            endpoint_override=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            scheme='https'
        )
        fs.get_file_info(bucket_name + "/")
        print("R2 connection successful.")
    except Exception as e:
        print(f"ERROR: Failed to connect or list R2 bucket '{bucket_name}': {e}")
        sys.exit(1)

    return bucket_name, fs

# Helper functions
# Reading only (ID/Sequence initially)
def load_data_from_parquet(r2_bucket, r2_dataset_path, r2_filesystem):
    """
        Loads the COMPLETE Parquet dataset from R2 into a Pandas DataFrame using PyArrow.
        Extracts sequence data (ID, Sequence) for embedding.
        Returns the sequence data list AND the full Pandas DataFrame.
    """
    full_dataset_uri = f"{r2_bucket}/{r2_dataset_path}"
    print(f"Loading full dataset into memory")
    
    try:
        print("Creating PyArrow ParquetDataset object...")
        dataset = pq.ParquetDataset(full_dataset_uri, filesystem=r2_filesystem)
        print(f"Dataset object created")

        # Columns needed
        all_columns = [field.name for field in DATA_SCHEMA]

        # Checking partitioning column in schema
        actual_schema_cols = dataset.schema.names
        columns_to_read = [col for col in all_columns if col in actual_schema_cols]
        if 'uniprot_id_prefix' in all_columns and 'uniprot_id_prefix' not in actual_schema_cols:
            print("Partitioning column not foudn in Parquet schema")

        print(f"Reading specified columns ({len(columns_to_read)}) into memory...")
        start_read = time.time()
        # Read the entier dataset into PyArrow table then convert into Pandas DF
        table_full = dataset.read(columns=columns_to_read)
        df_full_pd = table_full.to_pandas()
        
        del table_full
        gc.collect()
        end_read = time.time()
        print(f"Loaded full dataset into Pandas DataFrame with shape: {df_full_pd.shape}")
        print(f"Data loading took {end_read - start_read:.2f} seconds.")

        # Extracting only ID ans Sequence for embeddings
        print("Extracting uniprot_id and sequence columns for embedding...")
        if 'uniprot_id' not in df_full_pd.columns or 'sequence' not in df_full_pd.columns:
            print("ERROR: 'uniprot_id' or 'sequence' column not found in loaded data.")
            sys.exit(1)
        
        df_seq_pd = df_full_pd[['uniprot_id', 'sequence']].copy()

        # Validate and format Sequence Data
        sequence_data = []
        invalid_entries = 0
        for _, row in df_seq_pd.iterrows():
            item_id, sequence = row['uniprot_id'], row['sequence']
            if not isinstance(sequence, str):
                invalid_entries += 1
                continue
            sequence_cleaned = "".join(sequence.split()).upper()
            if not sequence_cleaned:
                invalid_entries += 1
                continue
            sequence_data.append((item_id, sequence_cleaned))
        
        processed_count = len(sequence_data)
        processed_count = len(sequence_data)
        if invalid_entries > 0: print(f"Skipped {invalid_entries} invalid/empty sequences.")
        if processed_count == 0: print("Error: No valid sequences found."); sys.exit(1)

        print(f"Prepared {processed_count} sequences for embedding.")

        return sequence_data, df_full_pd
    
    except MemoryError:
        print("Ran out of memory")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading Parquet data: {e}")
        sys.exit(1)

# ESM Model loading
def load_esm_model(model_name="facebook/esm2_t33_650M_UR50D"):
    """
        Loads ESM model and Tokenizer from Hugging Face Model Hub. 
    """
    print(f"Loading HF ESM model: {model_name}...")
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # Layer to use (last)
        layer_repr = model.config.num_hidden_layers

        # Device and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if device.type == "cuda":
            print(f"Working on GPU: {torch.cuda.get_device_name(0)}") 
        else:
            print("Working on CPU .") 

        print(f"ESM model '{model_name}' loaded successfully from Hugging Face.")
        print(f"Using output from layer {layer_repr} for embeddings.")
 
        return model, tokenizer, layer_repr, device
    except Exception as e:
        print(f"Error loading Hugging Face ESM model '{model_name}': {e}")
        exit(1)

# Embedding Generation
def generate_embeddings(model, tokenizer, sequence_data, batch_size=32, layer_repr=33, max_len=1022, device='cpu'):
    """
        Generate Embedding for Sequence using HF ESM Model
        Args:
            model: Loaded HF ESM model
            tokenizer: HF ESM tokenizer
            sequence_data: List of tuples [(id1, seq1), ...]
            batch_size: Number of sequences to process per batch
            layer_repr: The layer's representation to use
            max_len: Maximum sequence length (residues) BEFORE tokenization.
                     Sequences longer than this will have their embeddings calculated based
                     on the first `max_len` amino acids.
            device: torch device ('cuda' or 'cpu')
        Returns:
            Numpy array of embeddings (num_sequences, embed_dim)
            List of sequence IDs corresponding to the embedding rows
            List of IDs of sequences that were too long and got truncated
    """
    embeddings = []
    processed_ids = []
    truncated_ids = []
    num_sequences = len(sequence_data)
    embed_dim = model.config.hidden_size # Get embedding dimension from model config

    print(f"\nGenerating embeddings for {num_sequences} sequence(s) (batch size: {batch_size})...") # Corrected print
    print(f"Using representation from layer: {layer_repr}")
    print(f"Max seq length for embedding calculation: {max_len} residues")


    start_time = time.time()
    for i in range(0, num_sequences, batch_size):
        batch_data = sequence_data[i:i+batch_size]
        batch_ids = [item[0] for item in batch_data]
        batch_seqs_orig = [item[1] for item in batch_data]

        # Truncate seq if exceeds max len
        batch_seqs_processed = []
        current_truncated = []
        for seq_id, seq in zip(batch_ids, batch_seqs_orig):
            if len(seq) > max_len:
                batch_seqs_processed.append(seq[:max_len])
                current_truncated.append(seq_id)
            else:
                batch_seqs_processed.append(seq)

        if current_truncated:
            truncated_ids.extend(current_truncated)

        # Prepare batch for HF ESM model
        try:
            # Tokenize sequence (also on truncated sequence)
            inputs = tokenizer(
                batch_seqs_processed,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
        except Exception as e:
            print(f"\nError during tokenization (Batch {i//batch_size + 1}), skipping this batch.")
            print(f"Error: {e}")
            continue

        # Move tokenized inputs to the correct device
        inputs = {k : v.to(device) for k, v in inputs.items()} # Corrected variable name

        num_batches = (num_sequences + batch_size - 1) // batch_size
        print(f" Processing batch {i // batch_size + 1}/{num_batches}...", end='\r')

        # Model outputs
        with torch.no_grad():
            try:
                # Getting hidden states
                outputs = model(**inputs, output_hidden_states=True)
            except Exception as e:
                print(f"\nError during model inference (Batch {i//batch_size + 1}). Problematic IDs might be: {batch_ids}")
                print(f"Error: {e}")
                continue

        # Extract hidden states for the specified layer
        if not (0 <= layer_repr < len(outputs.hidden_states)):
            print(f"\nError: Invalid layer_repr index {layer_repr}. Max index is {len(outputs.hidden_states)-1}.")
            layer_repr = len(outputs.hidden_states) - 1 # Fallback to last layer
            print(f"Falling back to last layer index: {layer_repr}")

        token_representations = outputs.hidden_states[layer_repr] # (batch, seq_len_padded, embed_dim)

        # Loop through sequences in the batch to calculate per-sequence mean embeddings
        for j, (seq_id, seq) in enumerate(zip(batch_ids, batch_seqs_processed)):
            actual_seq_len = len(seq)
            # Indices: 0=[CLS], 1 to actual_seq_len = residues, actual_seq_len+1=[SEP], ... [PAD]
            seq_repr = token_representations[j, 1 : actual_seq_len + 1, :] # (actual_seq_len, embed_dim)

            if seq_repr.shape[0] == 0:
                print(f"Warning: Zero length sequence representation for ID '{seq_id}' after slicing. Using zero vector")
                seq_level_repr = torch.zeros(embed_dim, device=device)
            else:
                # Average pool over the actual residue representations
                seq_level_repr = seq_repr.mean(dim=0) # (embed_dim,)

            embeddings.append(seq_level_repr.cpu().numpy())
            processed_ids.append(seq_id)

        # Clean up GPU MEM
        del outputs, token_representations, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    end_time = time.time()

    print(f"\nEmbedding generation complete. Took {end_time - start_time:.2f} seconds. Processed {len(processed_ids)} sequences.")
    if truncated_ids:
         print(f"Warning: {len(truncated_ids)} sequences exceeded max length ({max_len}) and were truncated before tokenization.")

    return np.array(embeddings), processed_ids, truncated_ids


# Clustering the embeddings
def cluster_embeddings(embeddings, method, **kwargs):
    """
        Cluster embeddings using specified method
    """
    n_samples = embeddings.shape[0]
    if n_samples == 0:
        print("Error: No embeddings to cluster")
        exit(1)

    print(f"\n Clustering {n_samples} embeddings using {method}.") # Added method

    start_time = time.time()
    labels = None
    num_found_clusters = 0
    noise_points = 0 # Initialize noise points count

    if method == 'dbscan':
        eps = kwargs.get('eps', 0.5) # Use eps from kwargs, default 0.5
        min_samples = kwargs.get('min_samples', 5) # Use min_samples from kwargs, default 5
        print(f"  Using DBSCAN with eps={eps:.4f}, min_samples={min_samples}, metric='cosine'") # Use actual values
        if eps <= 0:
             print("Error: DBSCAN eps must be positive.")
             exit(1)
        if min_samples <= 0:
             print("Error: DBSCAN min_samples must be positive.")
             exit(1)
        try:
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1) # Pass correct params
            labels = clustering.fit_predict(embeddings)
            unique_labels = set(labels)
            num_found_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0) # Exclude noise points labeled as -1
            noise_points = np.sum(labels == -1) # Count noise points
            print(f"  DBSCAN clustering found {num_found_clusters} clusters and {noise_points} noise points.") # Report noise
        except Exception as e:
             print(f"Error during DBSCAN clustering: {e}")
             exit(1)
    else:
        print(f"Error: Unsupported clustering method '{method}'. Check --cluster_method argument.")
        exit(1)

    end_time = time.time()

    if labels is not None:
        print(f"Clustering complete. Took {end_time - start_time:.2f} seconds.")
        print(f"Assigned {num_found_clusters} unique cluster labels (excluding noise for DBSCAN).") # Clarified
        if noise_points > 0: # Report noise points if any
             print(f"  {noise_points} points were classified as noise by DBSCAN (label -1).")
    else:
        print("Clustering failed.")
        exit(1)

    return labels

# Data splitting
def split_data_by_clusters(processed_ids, cluster_labels, test_split_ratio, random_seed=42):

    if len(processed_ids) != len(cluster_labels): raise ValueError("Mismatch IDs vs labels.")

    valid_points = [(processed_ids[i], label) for i, label in enumerate(cluster_labels) if label != -1]
    num_noise_points = len(processed_ids) - len(valid_points)
    if num_noise_points > 0: print(f"\nNote: {num_noise_points} noise points excluded from split.")
    if not valid_points: print("Error: No valid points remaining."); return [], []

    clusters_by_label = {}
    for seq_id, label in valid_points:
        if label not in clusters_by_label: clusters_by_label[label] = []
        clusters_by_label[label].append(seq_id)

    cluster_list_of_ids = list(clusters_by_label.values())
    num_unique_clusters = len(cluster_list_of_ids)
    print(f"\nSplitting data based on {num_unique_clusters} clusters ({len(valid_points)} points)...")

    if num_unique_clusters < 2:
        print("Warning: Only <= 1 cluster. All valid data in training set.")
        train_ids = [seq_id for cluster in cluster_list_of_ids for seq_id in cluster]
        test_ids = []
        num_train_clusters, num_test_clusters = num_unique_clusters, 0
    else:
        train_clusters, test_clusters = train_test_split(
            cluster_list_of_ids, test_size=test_split_ratio, random_state=random_seed)
        train_ids = [seq_id for cluster in train_clusters for seq_id in cluster]
        test_ids = [seq_id for cluster in test_clusters for seq_id in cluster]
        num_train_clusters, num_test_clusters = len(train_clusters), len(test_clusters)

    print(f"Train set: {len(train_ids)} sequences from {num_train_clusters} clusters.")
    print(f"Test set:  {len(test_ids)} sequences from {num_test_clusters} clusters.")
    return train_ids, test_ids

# Saving subsets
def save_subset_to_parquet(id_list, df_full, r2_filesystem, r2_bucket, output_r2_path, schema=DATA_SCHEMA):
    """
    Filters the full Pandas DataFrame for the given IDs and saves the subset
    as a new Parquet dataset on R2 using PyArrow's `write_to_dataset`.
    """
    # Path within the bucket (key) 
    output_path_in_bucket = f"{r2_bucket}/{output_r2_path}" 
    output_uri_display = f"s3://{output_path_in_bucket}"

    if not id_list:
        print(f"Skipping save to {output_uri_display}: ID list is empty.")
        return
    if df_full is None or df_full.empty:
        print(f"Skipping save to {output_uri_display}: Full DataFrame is empty or None.")
        return

    print(f"\nPreparing to save {len(id_list)} entries to Parquet dataset: {output_uri_display}")
    print(f"  Filtering Pandas DataFrame...")
    start_filter_write = time.time()
    subset_table = None
    try:
        # 'uniprot_id' is string type
        id_list_str = [str(id_val) for id_val in id_list]
        subset_df = df_full[df_full['uniprot_id'].isin(id_list_str)].copy()

        if subset_df.empty:
            print(f"  Warning: Filtered subset for {output_uri_display} is empty. No data to save.")
            return

        print(f"  Filtered DataFrame shape: {subset_df.shape}")
        print(f"  Converting filtered Pandas subset to PyArrow Table...")

        # Add missing columns as None
        cols_to_add = []
        final_cols_order = [field.name for field in schema]
        for col_name in final_cols_order:
             if col_name not in subset_df.columns:
                subset_df[col_name] = None
                cols_to_add.append(col_name)
        # Reordering columns to match schema
        subset_df = subset_df[final_cols_order]
        if cols_to_add:
            print(f" Added missing columns: {cols_to_add}")

        subset_table = pa.Table.from_pandas(subset_df, schema=schema, preserve_index=False)
        del subset_df
        gc.collect()

        print(f"Writing subset PyArrow Table to Parquet at {output_uri_display}") # Use display URI here is fine
        pq.write_to_dataset(
            subset_table,
            root_path=output_path_in_bucket,
            schema=schema,
            filesystem=r2_filesystem,
            use_threads=True,
            existing_data_behavior='overwrite_or_ignore'
        )
        end_filter_write = time.time()
        print(f"  Parquet dataset saved successfully. Filtering and writing took {end_filter_write - start_filter_write:.2f}s.")

    except Exception as e:
        # Provide the display URI in the error message for clarity
        print(f"ERROR during subset filtering or Parquet writing for {output_uri_display}: {e}")
    finally:
        # Ensure table is deleted even on error
        if subset_table is not None:
            del subset_table
        gc.collect()
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Splits protein sequence data (read from R2 Parquet) into training/testing
        Parquet datasets based on ESM embedding clusters.
        """)

    # IO Arguments
    parser.add_argument('--r2_input_path', type=str, default=DEFAULT_R2_INPUT_DIR,
                        help=f'Path to the INPUT partitioned Parquet dataset directory within the R2 bucket. Default: {DEFAULT_R2_INPUT_DIR}')
    parser.add_argument('--r2_train_path', type=str, default=DEFAULT_R2_TRAIN_DIR,
                        help=f'Path for the OUTPUT training Parquet dataset directory within the R2 bucket. Default: {DEFAULT_R2_TRAIN_DIR}')
    parser.add_argument('--r2_test_path', type=str, default=DEFAULT_R2_TEST_DIR,
                        help=f'Path for the OUTPUT testing Parquet dataset directory within the R2 bucket. Default: {DEFAULT_R2_TEST_DIR}')
    parser.add_argument('--embeddings_out', type=str, default=None,
                        help='Optional: File path to save generated embeddings (numpy .npy format).')
    parser.add_argument('--ids_out', type=str, default=None,
                        help='Optional: File path to save list of IDs corresponding to embeddings (text file).')

    # ESM Arguments
    parser.add_argument('--esm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='HF ESM model identifier.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for ESM.')
    parser.add_argument('--max_len', type=int, default=1022, help='Max sequence length for ESM.')

    # Clustering Arguments
    parser.add_argument('--cluster_method', type=str, default='dbscan', choices=['dbscan', 'agglomerative'], help="Clustering algorithm.")
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps.')
    parser.add_argument('--min_samples', type=int, default=5, help='DBSCAN min_samples.')
    parser.add_argument('--distance_threshold', type=float, default=None, help='Agglomerative distance threshold.')
    parser.add_argument('--n_clusters', type=int, default=50, help='Agglomerative n_clusters.')

    # Splitting Arguments
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Proportion of clusters for test set.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting.')

    args = parser.parse_args()

    #  Input Validation 
    if not 0.0 < args.test_ratio < 1.0: print("Error: Test ratio invalid."); sys.exit(1)
    if args.batch_size <= 0: print("Error: batch_size invalid."); sys.exit(1)
    if args.max_len <= 0: print("Error: max_len invalid."); sys.exit(1)
    # Add detailed clustering param validation if needed

    #  Workflow 
    script_start_time = time.time()

    # Setup R2 connection
    r2_bucket, r2_filesystem = setup_r2_fs(ENV_FILE_PATH)

    # Load sequence data and get full Dask DataFrame
    sequence_data, df_full = load_data_from_parquet(r2_bucket, args.r2_input_path, r2_filesystem)
    if df_full is None:
        print("ERROR: Failed to load data, DataFrame is None.")
        sys.exit(1)

    # Load ESM Model
    model, tokenizer, layer_repr, device = load_esm_model(args.esm_model)

    # Generate Embeddings
    embeddings, processed_ids, _ = generate_embeddings(
        model, tokenizer, sequence_data, args.batch_size, layer_repr, args.max_len, device=device
    )
    if embeddings.shape[0] == 0: print("Error: No embeddings generated."); sys.exit(1)
    if len(processed_ids) != embeddings.shape[0]: print("Error: Mismatch embeddings vs IDs."); sys.exit(1)

    # Clean up model and sequence data from memory if possible
    del model, tokenizer, sequence_data
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save embeddings/IDs
    if args.embeddings_out: 
        print(f"Saving embeddings to {args.embeddings_out}...")
        np.save(args.embeddings_out, embeddings)
    if args.ids_out:
        print(f"Saving corresponding IDs to {args.ids_out}...")
        with open(args.ids_out, 'w') as f:
            for seq_id in processed_ids: f.write(str(seq_id) + '\n')

    # Cluster embeddings
    cluster_params = {}
    if args.cluster_method == 'dbscan':
        cluster_params['eps'] = args.eps
        cluster_params['min_samples'] = args.min_samples
    cluster_labels = cluster_embeddings(embeddings, method=args.cluster_method, **cluster_params)

    # Clean up embeddings
    del embeddings
    gc.collect()

    # Split IDs based on clusters
    train_ids, test_ids = split_data_by_clusters(processed_ids, cluster_labels, args.test_ratio, args.seed)

    # Clean up labels and IDs
    del cluster_labels, processed_ids
    gc.collect()

    # Save output datasets by filtering Dask DF and writing Parquet subsets
    save_subset_to_parquet(train_ids, df_full, r2_filesystem, r2_bucket, args.r2_train_path, schema=DATA_SCHEMA)
    save_subset_to_parquet(test_ids, df_full, r2_filesystem, r2_bucket, args.r2_test_path, schema=DATA_SCHEMA)

    # Final cleanup
    del df_full, train_ids, test_ids
    gc.collect()

    script_end_time = time.time()
    print("\nCompleted.")