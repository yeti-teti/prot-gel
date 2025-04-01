import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def collate_fn(batch):
    """
    Collate function for DataLoader. Pads sequences (using 0) and residue features.
    Stacks protein features and labels. Handles potential errors during batch processing.
    Filters out None items returned by dataset.__getitem__ on error.
    """

    # Filter out None items resulting from errors in __getitem__
    original_batch_len = len(batch)
    batch = [item for item in batch if item is not None]
    if not batch:
        print("Received empty batch")
        return None
    
    if len(batch) < original_batch_len:
        print(f"WARNING: collate_fn filtered out {original_batch_len - len(batch)} None items.")

    try:
        # Extract data from batch items
        sequences = [item['sequence'] for item in batch]
        residue_features = [item['residue_features'] for item in batch]
        protein_features = torch.stack([item['protein_features'] for item in batch])
        gelations = torch.stack([item['gelation'] for item in batch])
        uniprot_ids = [item['uniprot_id'] for item in batch]

        # Pad sequences using padding_idx=0
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        max_len = padded_sequences.shape[1]

        # Pad residue features
        # Calculating expected feature dimension form the first valid item if possible
        res_feature_dim = -1
        first_valid_res_feat = next((rf for rf in residue_features if rf.nelement() > 0 and len(rf.shape) > 1), None)
        if first_valid_res_feat is not None:
            res_feature_dim = first_valid_res_feat.shape[-1]
        else:
            res_feature_dim = 14 # 6 continuous + 8 SS classes
            print("Warning: Couldn't determine residue feature dim for batch")

        if res_feature_dim > 0:
            padded_residue_features = torch.zeros(len(batch), max_len, res_feature_dim, dtype=torch.float)
            for i, res_feat in enumerate(residue_features):
                if isinstance(res_feat, torch.Tensor) and res_feat.nelement() > 0 and len(res_feat.shape) == 2:
                    length = res_feat.shape[0]
                    current_dim = res_feat.shape[1]
                    if length > 0 and current_dim == res_feature_dim:
                        copy_len = min(length, max_len)
                        padded_residue_features[i, :copy_len, :] = res_feat[:copy_len, :]
                    elif length > 0:
                        print(f"Warning: Inconsistent feature dimensions")
                else:
                    print("Error: collate_rn failed to determine residue feature dimensions")
                    padded_residue_features = torch.zeros(len(batch), max_len, 14, dtype=torch.float)
        
        return {
            'sequence': padded_sequences,
            'residue_features': padded_residue_features,
            'protein_features': protein_features,
            'gelation': gelations.unsqueeze(1),  # Add channel dim for loss fn [batch, 1]
            'uniprot_ids': uniprot_ids
        }
    except Exception as e:
        # Error during collation, likely due to inconsistent data from __getitem__
        print(f"ERROR in collate_fn: {e}. Skipping batch.")
        return None