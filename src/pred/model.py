import torch
import torch.nn as nn

import math

class ProtProp(nn.Module):
    def __init__(self, embed_dim=64, num_layers=5, num_heads=6, max_len=2000, protein_encode_dim=32, dropout=0.4):
        super(ProtProp, self).__init__()

        # Sequence encoder (Protein sequences and residue features)
        # Token and residue embeddings
        self.embedding = nn.Embedding(22, embed_dim) # 20 aa + 1 unk + 1 padding (idx 0)
        residue_feature_dim = 14 # 6 continuous + 8 one-hot SS
        self.d_model = embed_dim + residue_feature_dim
        self.emb_dropout = nn.Dropout(dropout)

        # Positional Encoding
        pe = torch.zeros(1, max_len, self.d_model)            # [1, Lmax, D]
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=num_heads, 
            dim_feedforward= 4 * self.d_model, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
        total_dim = self.d_model + protein_encode_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(
        self,
        sequence: torch.Tensor,          # [B, L]
        residue_features: torch.Tensor,  # [B, L, 14]
        protein_features: torch.Tensor,  # [B, 13]
        padding_mask: torch.Tensor = None,  # [B, L]
    ) -> torch.Tensor:
        
        # Sequence encoder
        seq_embed = self.emb_dropout(self.embedding(sequence))  # [batch, seq_len, embed_dim]
        x = torch.cat([seq_embed, residue_features], dim=-1)  # [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]

        if padding_mask is not None:
            padding_mask = padding_mask.to(sequence.device).bool()

        encoder_out = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # Length-aware mean pooling
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1) # [B, L, 1] 
            seq_sum = (encoder_out * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            seq_repr = seq_sum / lengths
        else:
            seq_repr = encoder_out.mean(dim=1)

        # Protein feature encoder
        protein_out = self.protein_mlp(protein_features)  # [batch, protein_encode_dim]

        # Merge and classify
        combined = torch.cat([seq_repr, protein_out], dim=1)  # [batch, total_dim(D+P)]
        logits = self.classifier(combined)  # [batch, 1]
        return logits
    
    @staticmethod
    def load_from_checkpoint(path, config):
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        model = ProtProp(
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_len=config.max_len,
            protein_encode_dim=config.protein_encode_dim,
            dropout=config.dropout
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint