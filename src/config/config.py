import yaml
from dataclasses import dataclass

@dataclass
class Config:
    # Model hyperparameters
    embed_dim: int = 64
    num_layers: int = 3
    num_heads: int = 3
    protein_encode_dim: int = 32
    max_len: int = 2000
    dropout: float = 0.2
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 10
    lr: float = 0.001
    log_interval: int = 100
    num_workers: int = 10
    
    # Clustering
    n_clusters: int = 10
    train_frac: float = 0.8
    
def load_config(config_path):
    """Load configuraion from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)