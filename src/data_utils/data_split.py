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

import dask.dataframe as dd
import pyarrow.fs as pafs
from dotenv import load_dotenv

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.model_selection import train_test_split


# --- Configuration & R2 Setup ---
ENV_FILE_PATH = ".env"
# R2 Path for the INPUT dataset (MUST match db_writer.py output)
DEFAULT_R2_INPUT_DIR = "integrated_data/viridiplantae_dataset_partitioned"
# Default R2 Paths for the OUTPUT train/test Parquet datasets
DEFAULT_R2_TRAIN_DIR = "integrated_data/train_split_parquet"
DEFAULT_R2_TEST_DIR = "integrated_data/test_split_parquet"


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
    storage_options = {'key': access_key, 'secret': secret_key, 'endpoint_url': endpoint}
    
    try:
        fs = pafs.S3FileSystem(**storage_options)
        fs.ls(bucket_name)
        print("R2 connection successful.")
    except Exception as e:
        print(f"ERROR: Failed to connect or list R2 bucket '{bucket_name}': {e}")
        sys.exit(1)

    return bucket_name, storage_options

# Helper functions
# Reading only (ID/Sequence initially)
def load_data_from_parquet(r2_bucket, r2_dataset_path, storage_options):
    """
        Loads sequence data (ID, Sequence) from R2 Parquet using Dask for embedding.
        Also returns the lazy full Dask DataFrame for later subset saving.
    """
    full_dataset_uri = f"r2://{r2_bucket}/{r2_dataset_path}"
    print(f"Initializing Dask Dataframe from: {full_dataset_uri}")
    try:
        # Create the lazy Dask DataFrame for the full dataset
        # Read all columns eventually needed
        all_columns = ['uniprot_id', 'sequence', 'sequence_length', 'organism', 'taxonomy_id',
                       'physicochemical_properties', 'aa_composition', 'residue_features',
                       'structural_features', 'domains', 'gelation', 'uniprot_id_prefix']
        ddf_full = dd.read_parquet(full_dataset_uri, storage_options=storage_options, columns=all_columns)
        print(f"Full Dask Dataframe created with {ddf_full.npartitions} partitions.")

        # Extract only ID and sequence for embeddings
        print("Reading 'uniprot_id' and 'sequence' columns for embedding...")
        ddf_seq = ddf_full[['uniprot_id', 'sequence']].copy()
        df_seq_pd = ddf_seq.compute()
        print(f"Loaded {len(df_seq_pd)} sequences into memory")

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
        if invalid_entries > 0: print(f"Skipped {invalid_entries} invalid/empty sequences.")
        if processed_count == 0: print("Error: No valid sequences found."); sys.exit(1)

        print(f"Prepared {processed_count} sequences for embedding.")
        return sequence_data, ddf_full

    except ImportError: print("ERROR: Dask/PyArrow missing."); sys.exit(1)
    except Exception as e: print(f"ERROR loading Dask/Parquet: {e}"); sys.exit(1)

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
    # device = next(model.parameters()).device # Device is now passed explicitly
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
                 continue # Skip batch on error

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
def cluster_embeddings(embeddings, method, n_clusters=50, **kwargs):
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

    elif method == 'agglomerative':
        if n_samples <= 1:
             print("Only <=1 sample, cannot perform agglomerative clustering. Assigning cluster label 0.")
             return np.array([0]) if n_samples == 1 else np.array([])
        # Use distance_threshold or n_clusters, prioritize threshold if both given
        use_threshold = kwargs.get('distance_threshold') is not None
        if use_threshold:
            n_clusters_param = None
            threshold_param = kwargs['distance_threshold']
            print(f"  Using Agglomerative Clustering with distance_threshold={threshold_param:.4f} (cosine metric, average linkage)")
            if threshold_param <= 0:
                print("Error: distance_threshold must be positive for Agg Clustering.")
                exit(1)
        else:
            n_clusters_param = n_clusters # Use n_clusters passed to function
            threshold_param = None
            print(f"  Using Agglomerative Clustering with n_clusters={n_clusters_param} (cosine metric, average linkage)")
            if n_clusters_param <= 0:
                print("Error: n_clusters must be positive if not using distance_threshold.")
                exit(1)
            if n_clusters_param > n_samples:
                print(f"Warning: n_clusters ({n_clusters_param}) > n_samples ({n_samples}). Setting n_clusters to {n_samples}.")
                n_clusters_param = n_samples

        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters_param,
                distance_threshold=threshold_param,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            num_found_clusters = len(set(labels))
            print(f"  Agglomerative clustering found {num_found_clusters} clusters.")
        except Exception as e:
            print(f"Error during Agglomerative clustering: {e}")
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
    # (Same as previous version - Code omitted for brevity)
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
def save_subset_to_parquet(id_list, ddf_full, storage_options, r2_bucket, output_r2_path):
    """
    Filters the full Dask DataFrame for the given IDs and saves the subset
    as a new Parquet dataset on R2 using Dask's `to_parquet`.
    """
    output_uri = f"r2://{r2_bucket}/{output_r2_path}" # Use r2:// scheme

    if not id_list:
        print(f"Skipping save to {output_uri}: ID list is empty.")
        return

    print(f"\nPreparing to save {len(id_list)} entries to Parquet dataset: {output_uri}")
    print(f"  Filtering Dask DataFrame...")
    try:
        # Ensure IDs are strings for filtering if 'uniprot_id' is string type
        id_list_str = [str(id_val) for id_val in id_list]
        subset_ddf = ddf_full[ddf_full['uniprot_id'].isin(id_list_str)].copy()

        # Check if the filtered DataFrame is empty before writing
        has_data = len(subset_ddf.head(1)) > 0
        if not has_data:
             print(f"  Warning: Filtered subset for {output_uri} is empty. No data to save.")
             return

        # For parallel writing
        print(f"  Writing subset Parquet dataset to {output_uri}...")
        start_write = time.time()
        subset_ddf.to_parquet(
            output_uri,
            storage_options=storage_options,
            write_index=False, 
            overwrite=True,
        )
        end_write = time.time()
        print(f"  Parquet dataset saved successfully. Took {end_write - start_write:.2f}s.")

    except Exception as e:
        print(f"ERROR during subset filtering or Parquet writing for {output_uri}: {e}")
     

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

    # --- Input Validation ---
    if not 0.0 < args.test_ratio < 1.0: print("Error: Test ratio invalid."); sys.exit(1)
    if args.batch_size <= 0: print("Error: batch_size invalid."); sys.exit(1)
    if args.max_len <= 0: print("Error: max_len invalid."); sys.exit(1)
    # Add detailed clustering param validation if needed

    # --- Workflow ---
    # 0. Setup R2 connection
    r2_bucket, storage_options = setup_r2_fs(ENV_FILE_PATH)

    # 1. Load sequence data and get full Dask DataFrame
    sequence_data, ddf_full = load_data_from_parquet(r2_bucket, args.r2_input_path, storage_options)

    # 2. Load ESM Model
    model, tokenizer, layer_repr, device = load_esm_model(args.esm_model)

    # 3. Generate Embeddings
    embeddings, processed_ids, _ = generate_embeddings(
        model, tokenizer, sequence_data, args.batch_size, layer_repr, args.max_len, device=device
    )
    if embeddings.shape[0] == 0: print("Error: No embeddings generated."); sys.exit(1)
    if len(processed_ids) != embeddings.shape[0]: print("Error: Mismatch embeddings vs IDs."); sys.exit(1)

    # 4. Save embeddings/IDs (Optional)
    if args.embeddings_out: print(f"Saving embeddings to {args.embeddings_out}..."); np.save(args.embeddings_out, embeddings)
    if args.ids_out:
        print(f"Saving corresponding IDs to {args.ids_out}...")
        with open(args.ids_out, 'w') as f:
            for seq_id in processed_ids: f.write(str(seq_id) + '\n')

    # 5. Cluster embeddings
    cluster_params = {}
    if args.cluster_method == 'agglomerative':
        cluster_params['n_clusters'] = args.n_clusters; cluster_params['distance_threshold'] = args.distance_threshold
    elif args.cluster_method == 'dbscan':
        cluster_params['eps'] = args.eps; cluster_params['min_samples'] = args.min_samples
    cluster_labels = cluster_embeddings(embeddings, method=args.cluster_method, **cluster_params)

    # 6. Split IDs based on clusters
    train_ids, test_ids = split_data_by_clusters(processed_ids, cluster_labels, args.test_ratio, args.seed)

    # 7. Save output datasets by filtering Dask DF and writing Parquet subsets
    # Ensure ddf_full is available here
    save_subset_to_parquet(train_ids, ddf_full, storage_options, r2_bucket, args.r2_train_path)
    save_subset_to_parquet(test_ids, ddf_full, storage_options, r2_bucket, args.r2_test_path)

    print("\nCompleted.")