import yaml
from dataclasses import dataclass

@dataclass
class Config:
    # Model hyperparameters
    embed_dim: int = 64
    num_filters: int = 64
    kernel_sizes: list = None  # Will be set to [3, 5, 7] by default
    protein_encode_dim: int = 32
    dropout: float = 0.2
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 10
    lr: float = 0.001
    log_interval: int = 100
    
    # Clustering
    n_clusters: int = 10
    train_frac: float = 0.8

    def __post_init__(self):
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 7]
    
def load_config(config_path):
    """Load configuraion from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)