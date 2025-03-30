import json
import argparse
import gc
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Import necessary clustering algorithms
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
# KMeans is not imported as it's not an option in the final argparse
from sklearn.model_selection import train_test_split

# Helper functions
def load_data_from_json(json_filepath):
    """
        Load data from file.

        Returns:
            - sequence data: List of tuples [(id1, seq1), ...]
            - original_data_map: Dict mapping ID -> original feature dictionary
    """
    print(f"Loading data from {json_filepath}") # Corrected typo
    try:
        with open(json_filepath, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("JSON data should be a dictionary")

        sequence_data = []
        original_data_map = {}
        count = 0
        invalid_entries = 0

        # Iterate through the IDs
        for item_id, item_data in data.items():
            count += 1
            if not isinstance(item_data, dict):
                print(f"Warning: Value for ID '{item_id}' is not dict, skipping")
                invalid_entries += 1
                continue
            if 'sequence' not in item_data:
                print(f"Warning: Sequence not present for '{item_id}', skipping")
                invalid_entries += 1
                continue

            sequence = item_data['sequence']
            if not isinstance(sequence, str):
                print(f"Warning: Invalid sequence type (not string) for ID: '{item_id}', skipping") 
                invalid_entries += 1
                continue

            sequence_cleaned = "".join(sequence.split()).upper()
            if not sequence_cleaned:
                print(f"Warning: Empty sequence after cleaning for ID '{item_id}', skipping") 
                invalid_entries += 1
                continue

            # Uniprot ID as identifier
            sequence_data.append((item_id, sequence_cleaned))
            # Store the original feature dictionary
            original_data_map[item_id] = item_data 

        processed_count = len(sequence_data)
        print(f"Processed {count} entries from JSON")
        if invalid_entries > 0:
            print(f"Skipped {invalid_entries} invalid entries")
        if processed_count == 0:
            print("Error: No valid sequence entries found") 
            exit(1)

        print(f"Successfully loaded {processed_count} valid sequence entries")
        return sequence_data, original_data_map

    except FileNotFoundError:
        print(f"Error: Input file not found at {json_filepath}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
        exit(1)
    except ValueError as e:
        print(f"Error: Data format validation failed - {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        exit(1)

def save_data_dict_to_json(data_dict, json_filepath):
    """
        Saves a dict (ID -> features) to JSON file.
    """
    print(f"Saving {len(data_dict)} entries to {json_filepath}...") 
    try:
        with open(json_filepath, 'w') as f:
            json.dump(data_dict, f, indent=4)
        print("Saved complete.") 
    except Exception as e:
        print(f"Error: Could not save data to {json_filepath}: {e}")
        exit(1)

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
def split_data_by_clusters(processed_ids, original_data_map, cluster_labels, test_split_ratio, random_seed=42):
    """
        Splits the original data (from original_data_map) based on clustering assignments
        Returns train_data_dict and test_data_dict (dict mapping ID -> features)
    """
    if len(processed_ids) != len(cluster_labels):
        raise ValueError("Mismatch between number of processed IDs and cluster labels.")

    # Filter out noise points (label -1) before splitting clusters
    valid_indices = [i for i, label in enumerate(cluster_labels) if label != -1]
    valid_labels = [cluster_labels[i] for i in valid_indices]
    # valid_processed_ids = [processed_ids[i] for i in valid_indices] # Not needed directly

    num_valid_points = len(valid_indices)
    num_noise_points = len(processed_ids) - num_valid_points

    if num_noise_points > 0:
        print(f"\nNote: {num_noise_points} noise points (label -1) were excluded from train/test split.")

    if num_valid_points == 0:
        print("Error: No valid points remaining after excluding noise. Cannot split.")
        return {}, {} # Return empty dicts


    num_unique_labels = len(set(valid_labels))
    print(f"\n Splitting data based on {num_unique_labels} clusters (using {num_valid_points} non-noise points)...")

    # Group valid original indices by their cluster label
    clusters_by_label = {}
    for i, label in enumerate(valid_labels): # Iterate through valid labels
        original_index = valid_indices[i]    # Get the original index corresponding to this valid point
        if label not in clusters_by_label:
            clusters_by_label[label] = []
        clusters_by_label[label].append(original_index) # Store ORIGINAL index from processed_ids

    # Get list of clusters, where each cluster contains original indices from processed_ids/embeddings
    cluster_list_of_indices = list(clusters_by_label.values())

    if len(cluster_list_of_indices) < 2:
        print("Warning: Only <= 1 cluster found among non-noise points. Cannot split into train/test. Returning all valid data as training set.")
        train_indices_flat = [idx for cluster in cluster_list_of_indices for idx in cluster]
        test_indices_flat = []
        num_train_clusters = len(cluster_list_of_indices)
        num_test_clusters = 0
    else:
        # Split the clusters
        train_clusters, test_clusters = train_test_split(
            cluster_list_of_indices,
            test_size=test_split_ratio,
            random_state=random_seed
        )
        # Flatten lists to get original indices relative to the processed_ids list
        train_indices_flat = [idx for cluster in train_clusters for idx in cluster]
        test_indices_flat = [idx for cluster in test_clusters for idx in cluster]
        num_train_clusters = len(train_clusters)
        num_test_clusters = len(test_clusters)

    # Output dictionaries
    train_data_dict = {}
    test_data_dict = {}
    missing_in_original = 0

    # Adding to train dict
    for idx in train_indices_flat:
        if idx >= len(processed_ids): # Sanity check against original list length
            print(f"Warning: Train index {idx} out of bounds for processed_ids (len {len(processed_ids)}). Skipping.") # Corrected print
            continue
        seq_id = processed_ids[idx]
        original_item = original_data_map.get(seq_id)
        if original_item:
               train_data_dict[seq_id] = original_item
        else:
            missing_in_original += 1
            print(f"Warning: ID '{seq_id}' from train split not found in original_data_map.")

    # Adding to test dict
    for idx in test_indices_flat:
        if idx >= len(processed_ids): # Sanity check
            print(f"Warning: Test index {idx} out of bounds for processed_ids (len {len(processed_ids)}). Skipping.") # Corrected print
            continue
        seq_id = processed_ids[idx]
        original_item = original_data_map.get(seq_id)
        if original_item:
            test_data_dict[seq_id] = original_item
        else:
            missing_in_original += 1
            print(f"Warning: ID '{seq_id}' from test split not found in original_data_map.")

    if missing_in_original > 0:
        print(f"Total Warning Count: {missing_in_original} IDs from clustering step were not found in the original data map during split assignment.")

    print(f"Train set: {len(train_data_dict)} sequences from {num_train_clusters} clusters.")
    print(f"Test set:  {len(test_data_dict)} sequences from {num_test_clusters} clusters.") # Corrected spacing

    return train_data_dict, test_data_dict # Return dictionaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Splits protein sequence data (from integrated JSON dictionary) into training
        and testing sets based on ESM embedding clusters using Hugging Face Transformers.
        """) 

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input integrated JSON file (dictionary: ID -> features).')
    parser.add_argument('--train_out', type=str, default='train_set_integrated.json',
                        help='Output file for the training set (JSON dictionary). Default: train_set_integrated.json')
    parser.add_argument('--test_out', type=str, default='test_set_integrated.json',
                        help='Output file for the testing set (JSON dictionary). Default: test_set_integrated.json')
    parser.add_argument('--esm_model', type=str, default='facebook/esm2_t33_650M_UR50D',
                        help='Hugging Face identifier of the pre-trained ESM model to use (e.g., facebook/esm2_t33_650M_UR50D). Default: facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for ESM embedding generation. Adjust based on GPU memory. Default: 32')
    parser.add_argument('--max_len', type=int, default=1022,
                         help='Maximum sequence length (residues) BEFORE tokenization. Longer sequences truncated. Default: 1022')


    # Clustering Args
    parser.add_argument('--cluster_method', type=str, default='dbscan', choices=['dbscan', 'agglomerative'],
                        help="Clustering algorithm ('dbscan', 'agglomerative'). Default: dbscan")
    parser.add_argument('--eps', type=float, default=0.5, 
                        help='DBSCAN eps parameter (max distance between samples for neighborhood). Needs tuning. Default: 0.5')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='DBSCAN min_samples parameter (number of samples in neighborhood). Default: 5')
    parser.add_argument('--distance_threshold', type=float, default=None,
                         help='Distance threshold ONLY for Agglomerative clustering (overrides --n_clusters if set). Uses cosine distance. Try values like 0.6-1.0.')
    parser.add_argument('--n_clusters', type=int, default=50,
                        help='Number of clusters target ONLY for AgglomerativeClustering when distance_threshold is not set. Default: 50')

    # Common args
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Proportion of non-noise clusters to allocate to the test set. Default: 0.2')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split reproducibility. Default: 42') # Removed mention of clustering seed
    parser.add_argument('--embeddings_out', type=str, default=None,
                        help='Optional: File path to save the generated embeddings (numpy .npy format).')
    parser.add_argument('--ids_out', type=str, default=None,
                         help='Optional: File path to save the list of IDs corresponding to the embeddings (text file).')

    args = parser.parse_args()

    # Input validation
    if not 0.0 < args.test_ratio < 1.0:
        print("Error: Test split ratio must be between 0.0 and 1.0 (exclusive)")
        exit(1)
    # Clustering specific validation
    if args.cluster_method == 'agglomerative':
        if args.distance_threshold is None and args.n_clusters <= 0:
            print("Error: For Agglomerative, must provide positive --n_clusters OR positive --distance_threshold.")
            exit(1)
        if args.distance_threshold is not None and args.distance_threshold <= 0:
             print("Error: --distance_threshold must be positive for Agglomerative clustering.")
             exit(1)
    # DBSCAN validation
    if args.cluster_method == 'dbscan':
        if args.eps <= 0:
            print("Error: --eps must be positive for DBSCAN.")
            exit(1)
        if args.min_samples <= 0:
            print("Error: --min_samples must be positive for DBSCAN.")
            exit(1)

    # Other validation
    if args.batch_size <= 0:
         print("Error: batch_size must be positive.")
         exit(1)
    if args.max_len <= 0:
         print("Error: max_len must be positive.")
         exit(1)

    # 1. Load data
    sequence_data, original_data_map = load_data_from_json(args.input)

    # 2. Load ESM Model
    model, tokenizer, layer_repr, device = load_esm_model(args.esm_model)

    # 3. Generate Embeddings
    embeddings, processed_ids, _ = generate_embeddings(
        model, tokenizer, sequence_data, args.batch_size, layer_repr, args.max_len, device=device
    )

    if embeddings.shape[0] == 0:
         print("Error: No embeddings were generated. Exiting.")
         exit(1)
    if len(processed_ids) != embeddings.shape[0]:
         print(f"Error: Mismatch between number of embeddings ({embeddings.shape[0]}) and processed IDs ({len(processed_ids)}).")
         exit(1)

    # 4. Save embeddings and IDs
    if args.embeddings_out:
        print(f"Saving embeddings to {args.embeddings_out}...")
        np.save(args.embeddings_out, embeddings)
    if args.ids_out:
         print(f"Saving corresponding IDs to {args.ids_out}...")
         with open(args.ids_out, 'w') as f:
              for seq_id in processed_ids:
                   f.write(str(seq_id) + '\n')

    # 5. Cluster embeddings
    cluster_params = {}
    if args.cluster_method == 'agglomerative':
        # Pass n_clusters and distance_threshold, let the function prioritize
        cluster_params['n_clusters'] = args.n_clusters
        cluster_params['distance_threshold'] = args.distance_threshold
    elif args.cluster_method == 'dbscan':
        cluster_params['eps'] = args.eps
        cluster_params['min_samples'] = args.min_samples

    cluster_labels = cluster_embeddings(
        embeddings,
        method=args.cluster_method,
        **cluster_params
    )

    # 6. Split data based on clusters
    train_data_dict, test_data_dict = split_data_by_clusters(
        processed_ids,           # List of IDs corresponding to embeddings
        original_data_map,       # Map ID -> original dict
        cluster_labels,          # Labels from clustering step
        args.test_ratio,
        args.seed                # Seed for train/test split consistency
    )

    # 7. Save output datasets
    save_data_dict_to_json(train_data_dict, args.train_out)
    save_data_dict_to_json(test_data_dict, args.test_out)

    print("\nCompleted.")