import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Collate function for DataLoader. Pads sequences (using 0) and residue features.
    Stacks protein features and labels. Handles potential errors during batch processing.
    Filters out None items returned by dataset.__getitem__ on error.
    """
    # Filter out None items resulting from errors in __getitem__
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch failed

    try:
        sequences = [item['sequence'] for item in batch]
        residue_features = [item['residue_features'] for item in batch]
        protein_features = torch.stack([item['protein_features'] for item in batch])
        gelations = torch.stack([item['gelation'] for item in batch])
        # uniprot_ids = [item['uniprot_id'] for item in batch] # Optional

        # Pad sequences using padding_idx=0
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

        # Pad residue features
        if residue_features and residue_features[0].nelement() > 0:
             # Check if feature dimension is consistent if possible
             res_feature_dim = residue_features[0].shape[-1] # Get from last dim
             max_len = padded_sequences.shape[1]
             padded_residue_features = torch.zeros(len(batch), max_len, res_feature_dim, dtype=torch.float)
             for i, res_feat in enumerate(residue_features):
                 length = res_feat.shape[0]
                 if length > 0:
                     # Ensure feature dim matches before assignment
                     if res_feat.shape[-1] == res_feature_dim:
                         padded_residue_features[i, :length, :] = res_feat
                     else:
                         print(f"Warning: collate_fn found inconsistent residue feature dimension (expected {res_feature_dim}, got {res_feat.shape[-1]}). Padding with zeros.")

        else:
            # Fallback: create empty or zero tensor if no residue features present
            print("Warning: collate_fn creating zero tensor for residue features (likely due to upstream error or empty batch).")
            # Determine expected feature dim requires access to mean_std or config
            num_residue_features_expected = 6 + 8 # 6 continuous + 8 SS classes (example) - HARDCODED, BAD!
            padded_residue_features = torch.zeros(len(batch), padded_sequences.shape[1], num_residue_features_expected, dtype=torch.float)


        return {
            'sequence': padded_sequences,
            'residue_features': padded_residue_features,
            'protein_features': protein_features,
            'gelation': gelations.unsqueeze(1) # Add channel dim for loss function
        }
    except Exception as e:
        # Error during collation, likely due to inconsistent data from __getitem__
        print(f"ERROR in collate_fn: {e}. Skipping batch.")
        return None