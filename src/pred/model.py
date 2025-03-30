import torch
import torch.nn as nn

class ProtProp(nn.Module):
    def __init__(self, embed_dim=64, num_filters=64, kernel_sizes=[3,5,7], protein_encode_dim=32, dropout=0.4):
        super(ProtProp, self).__init__()

        # Sequence encoder
        self.embedding = nn.Embedding(21, embed_dim) # 20 aa + 1 unk
        residue_feature_dim = 14 # 6 continuous + 8 one-hot SS
        input_dim = embed_dim + residue_feature_dim
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, num_filters, k, padding=0),
                nn.ReLU(),
                nn.BatchNorm1d(num_filters)
            ) for k in kernel_sizes
        ])

        # Protein feature encoder
        protein_input_dim = 13
        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, protein_encode_dim),
            nn.ReLU(),
            nn.BatchNorm1d(protein_encode_dim)
        )

        # Classifier
        total_dim = num_filters * len(kernel_sizes) + protein_encode_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, sequence, residue_features, protein_features):
        # Sequence encoder
        seq_embed = self.embedding(sequence)  # [batch, seq_len, embed_dim]
        x = torch.cat([seq_embed, residue_features], dim=-1)  # [batch, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        conv_outputs = [conv(x) for conv in self.convs]  # List of [batch, num_filters, seq_len']
        pooled = [torch.max(out, dim=2)[0] for out in conv_outputs]  # List of [batch, num_filters]
        cnn_out = torch.cat(pooled, dim=1)  # [batch, num_filters * len(kernel_sizes)]

        # Protein feature encoder
        protein_out = self.protein_mlp(protein_features)  # [batch, protein_encode_dim]

        # Merge and classify
        combined = torch.cat([cnn_out, protein_out], dim=1)  # [batch, total_dim]
        logits = self.classifier(combined)  # [batch, 1]
        return logits
    
    def load_from_checkpoint(path, config):
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        model = ProtProp(
            embed_dim=config.embed_dim,
            num_filters=config.num_filters,
            kernel_sizes=config.kernel_sizes,
            protein_encode_dim=config.protein_encode_dim,
            dropout=config.dropout
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint