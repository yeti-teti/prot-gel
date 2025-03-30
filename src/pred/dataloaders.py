import torch

def collate_fn(batch):
    """Pad sequences and residue features to the maximum length in the batch."""
    sequences = [item['sequence'] for item in batch]
    residue_features = [item['residue_features'] for item in batch]
    protein_features = torch.stack([item['protein_features'] for item in batch])
    gelations = torch.stack([item['gelation'] for item in batch])

    max_len = max(len(seq) for seq in sequences)
    padded_sequences = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_residue_features = torch.zeros(len(batch), max_len, residue_features[0].shape[1], dtype=torch.float)

    for i, (seq, res_feat) in enumerate(zip(sequences, residue_features)):
        length = len(seq)
        padded_sequences[i, :length] = seq
        padded_residue_features[i, :length, :] = res_feat

    return {
        'sequence': padded_sequences,
        'residue_features': padded_residue_features,
        'protein_features': protein_features,
        'gelation': gelations
    }


