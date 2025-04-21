# Prot-GEL

## Overview

This repository implements the computational aspects of a machine learning system for predicting protein gelation properties based on structural and physicochemical features. The implementation draws from concepts described in patent US20240016179A1("SELECTING FOOD INGREDIENTS FROM VECTOR REPRESENTATIONS OF INDIVIDUAL PROTEINS USING CLUSTER ANALYSIS AND PRECISION FERMENTATION").

## Data Processing Pipeline

### 1. Data Collection and Integration

The system begins with a comprehensive data collection process that extracts protein information from multiple sources:

1. **Primary Database Sources** (handled by `db_writer.py`):
    - **UniProt/SwissProt**: Provides protein sequence data, taxonomy information, and basic annotations
    - **PDB (Protein Data Bank)**: Supplies 3D structural information for proteins with resolved structures
    - **AlphaFold Database**: Provides predicted 3D structures for proteins
    - **PFAM**: Offers domain annotations and functional classifications
2. **Data Integration Process**:
    - Protein sequences are extracted from UniProt files
    - Physicochemical properties are calculated for each protein (molecular weight, aromaticity, instability index, etc.)
    - Residue-level features are computed for each amino acid position (hydrophobicity, polarity, volume)
    - PDB structures are processed to extract secondary structure elements (alpha helix, beta sheet, coil percentages)
    - DSSP is run on structure files to determine additional properties like accessibility and backbone angles
    - Domain information is extracted from PFAM HMM scans
    - Gelation prediction labels are assigned based on predefined criteria
3. **Integrated JSON Creation**:
    - All extracted and computed data is merged into a comprehensive JSON structure
    - Each protein is represented by its UniProt ID as the key and a nested dictionary of features
    - The integrated data is stored as `integrated_data.json`

### 2. Cloud Storage Upload

The integrated JSON data is then transformed and uploaded to Cloudflare R2 storage (handled by `db_writer_cloud.py`):

1. **Parquet Conversion**:
    - JSON data is loaded and converted to a Pandas DataFrame
    - A partitioning column `uniprot_id_prefix` is added based on the first character of each protein's UniProt ID
    - The DataFrame is converted to a PyArrow Table with a defined schema
2. **R2 Storage Upload**:
    - A connection to Cloudflare R2 is established using credentials from `.env`
    - The PyArrow Table is written to R2 as a partitioned Parquet dataset
    - Data is partitioned by the `uniprot_id_prefix` column for efficient querying
    - The final dataset is stored at the configured path in the R2 bucket (e.g., `integrated_data/viridiplantae_dataset_partitioned_from_json`)
3. **Data Verification**:
    - The uploaded dataset can be verified using `db_r2_check.py`
    - This script samples records from the Parquet dataset to confirm successful upload and correct schema

### 3. Dataset Splitting

The uploaded Parquet dataset is then split into training and testing subsets (handled by `data_split.py`):

1. **Protein Embedding Generation**:
    - Protein sequences are loaded from the Parquet dataset
    - Sequences are embedded using the ESM protein language model
    - These embeddings capture semantic information about protein structure and function
2. **Clustering for Stratified Splitting**:
    - DBSCAN clustering is applied to group proteins by similarity
    - This ensures proteins with similar properties are kept together in either training or testing
    - Clusters are formed based on an adjustable similarity threshold (eps parameter)
3. **Train/Test Dataset Creation**:
    - Clusters are randomly assigned to either training or testing sets
    - Data is filtered accordingly and written back to R2 as separate Parquet datasets
    - The resulting datasets are stored at configured paths (e.g., `integrated_data/train_split_parquet` and `integrated_data/test_split_parquet`)

## Technical Implementation

### Model Architecture

The core model (`ProtProp`) is a transformer-based architecture that processes:

- Amino acid sequences (embedded)
- Residue-level features (hydrophobicity, polarity, volume, etc.)
- Protein-level features (molecular weight, charge, structural percentages, etc.)

The model combines these features through:

- Token embedding
- Residue feature concatenation
- Positional encoding
- Multi-head self-attention
- Feed-forward neural networks

### Model Architecture Details

The core model (`ProtProp`) contains:

- **Embedding Layer**: Maps amino acid tokens to a learned 64-dimensional vector space
- **Positional Encoding**: Adds sinusoidal position information to handle sequence order
- **Transformer Encoder**: Multi-head self-attention with 3 attention heads and 3 encoder layers
- **Protein Feature Encoder**: Two-layer MLP with batch normalization for global protein features
- **Feature Fusion**: Concatenation of sequence-derived and protein-level features
- **Classification Head**: Two-layer MLP with dropout for final prediction

The model processes proteins with the following information flows:

1. Sequence data → Embedding → Feature concatenation → Positional encoding → Transformer
2. Protein features → MLP → Feature normalization
3. Combined features → Classification head → Gelation probability

## Code Structure

The repository is organized into several key components:

```
protein-gelation-prediction/
├── config/                      # Configuration files
│   └── config.yaml              # Hyperparameter settings
├── pred/                        # Core prediction modules
│   ├── __init__.py              # Package initialization
│   ├── dataset.py               # Dataset loading and preprocessing
│   ├── dataloaders.py           # Batch collation and processing
│   ├── model.py                 # Neural network architecture
│   └── model_runner.py          # Training and inference pipeline
├── utils/                       # Utility scripts
│   ├── calculate_stats.py       # Feature normalization statistics
│   ├── data_pdb_parse.py        # PDB file processing
│   ├── data_split.py            # Train/test splitting utilities
│   ├── db_r2_check.py           # Database connectivity testing
│   ├── db_writer.py             # Database population
│   ├── db_writer_cloud.py       # Cloud database utilities
│   └── extract_af_pdbs.py       # AlphaFold PDB utilities
├── main.py                      # Command-line interface
├── run.sh                       # Example execution scripts
├── requirements.txt             # Dependencies
└── .env.example                 # Environment variable template
```

### Key Files in Detail

- **db_writer.py**: Handles the collection and integration of protein data:
    - Parses UniProt files to extract protein sequences and metadata
    - Calculates physicochemical properties using BioPython
    - Processes PDB files to extract structural information
    - Runs DSSP to obtain secondary structure and residue properties
    - Conducts HMMSCAN to find protein domains
    - Integrates all data into a comprehensive JSON structure
- **db_writer_cloud.py**: Manages the transformation and upload of data to cloud storage:
    - Converts JSON data to PyArrow tables
    - Adds partitioning information
    - Connects to Cloudflare R2 using authentication credentials
    - Uploads partitioned Parquet data to the R2 bucket
- **data_split.py**: Handles train/test splitting:
    - Generates protein embeddings using ESM models
    - Clusters proteins using DBSCAN
    - Ensures similar proteins are kept in the same split
    - Creates Parquet datasets for training and testing
- **config.py**: Defines a `Config` dataclass that stores model hyperparameters and training settings, along with a function to load configuration from YAML.
- **dataset.py**: Implements the `ProteinDataset` class that:
    - Connects to R2 storage
    - Loads protein data from Parquet files
    - Implements caching for efficient data access
    - Processes and normalizes protein features
    - Handles sequence encoding and feature extraction
- **dataloaders.py**: Contains the `collate_fn` function that:
    - Batches variable-length protein sequences
    - Handles padding and masking
    - Manages error cases in batch processing
    - Combines heterogeneous feature types
- **model.py**: Defines the `ProtProp` neural network that:
    - Embeds protein sequences
    - Processes residue-level features
    - Implements transformer architecture
    - Handles padding masks for variable-length sequences
    - Fuses different feature types for final prediction
- **model_runner.py**: Implements the `ModelRunner` class that:
    - Manages the training loop
    - Handles validation and testing
    - Implements checkpointing
    - Provides inference capabilities
    - Reports metrics and progress
- **calculate_stats.py**: Computes normalization statistics:
    - Calculates mean and standard deviation for all features
    - Handles missing values and outliers
    - Generates statistics for both protein and residue features

## Installation and Setup

### Environment Setup

```
# Clone the repository
git clone https://github.com/yourusername/protein-gelation-prediction.git
cd protein-gelation-prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables for R2 access
cp .env.example .env
# Edit .env to include:
# CLOUDFARE_ACCESS_KEY=your_access_key
# CLOUDFARE_SECRET_KEY=your_secret_key
# CLOUDFARE_ACCOUNT_ID=your_account_id
# CLOUDFARE_BUCKET_NAME=your_bucket_name
# CLOUDFARE_ENDPOINT=https://your_endpoint_url
```

### Data Preparation Pipeline

1. **Data Collection and Integration**:
    
    ```
    # Process UniProt, PDB, and domain data
    python db_writer.py
    ```
    
2. **Upload to Cloudflare R2**:

```
# Convert JSON to Parquet and upload to R2
python db_writer_cloud.py
```

1. **Verify Data Upload**:

```
# Check if data was uploaded correctly
python db_r2_check.py
```

1. **Statistics Calculation:**

```
# Calculate normalization statistics for features
python calculate_stats.py --r2_input_path integrated_data/viridiplantae_dataset_partitioned_from_json --output_file data/mean_std.json
```

1. **Dataset Splitting**:

```
# Split dataset into training and testing sets
python data_split.py \
  --esm_model facebook/esm2_t12_35M_UR50D \
  --cluster_method dbscan \
  --eps 0.01 \
  --min_samples 10 \
  --r2_input_path integrated_data/viridiplantae_dataset_partitioned_from_json \
  --r2_train_path integrated_data/train_split_parquet \
  --r2_test_path integrated_data/test_split_parquet
```

## Usage

### Model Training

```
python main.py train \
    --config config/config.yaml \
    --mean_std_json_path data/mean_std.json \
    --r2_env_path .env \
    --output_dir results/training_run_$(date +%Y%m%d_%H%M%S) \
    --train_r2_path integrated_data/train_split_parquet \
    --val_r2_path integrated_data/test_split_parquet \
    --verbosity info
```

Training options:

- `-config`: Path to configuration YAML file
- `-mean_std_json_path`: Path to normalization statistics
- `-output_dir`: Directory for saving checkpoints and logs
- `-train_r2_path`: R2 path to training dataset
- `-val_r2_path`: R2 path to validation dataset
- `-model`: Optional path to resume training from checkpoint
- `-verbosity`: Logging level (debug, info, warning, error)

### Model Evaluation

```
python main.py evaluate \
    --config config/config.yaml \
    --mean_std_json_path data/mean_std.json \
    --r2_env_path .env \
    --output_dir results/evaluation \
    --test_r2_path integrated_data/test_split_parquet \
    --model results/training_run_20250402_202118/epoch=1-step=543.ckpt \
    --results_file evaluation_results.txt
```

Evaluation options:

- `-test_r2_path`: R2 path to test dataset
- `-model`: Path to trained model checkpoint
- `-results_file`: Filename for evaluation metrics output

### Prediction on New Data

```
python main.py predict \
    --config config/config.yaml \
    --mean_std_json_path data/mean_std.json \
    --r2_env_path .env \
    --output_dir results/predictions \
    --predict_r2_path integrated_data/predict_input_parquet \
    --model results/training_run_20250402_202118/epoch=1-step=543.ckpt \
    --results_file predictions.tsv
```

Prediction options:

- `-predict_r2_path`: R2 path to dataset for prediction
- `-results_file`: Output file for predictions (TSV format)

## Comparison to Patent US 2024/0016179 A1

This implementation focuses specifically on the computational aspects of the patent's broader discovery system. The patent describes a comprehensive "flywheel" approach (as shown in Figure 1) that combines in silico prediction with empirical testing in an iterative learning cycle.

### Key Components Implemented

1. **Data Processing Pipeline** (Patent Fig. 2):
    - Implementation of database integration from multiple sources
    - Processing and feature extraction from sequence and structure data
    - Storage in cloud-based partitioned datasets
2. **Vector Representation** (Patent Fig. 4A):
    - Implementation of protein encoding through sequence, residue features, and protein features
    - Feature normalization and discretization
    - Vector-based protein representation
3. **Predictive Modeling** (Patent Fig. 4B):
    - Machine learning (transformer-based architecture)
    - Deep learning (self-attention and feature fusion)
    - Ensembling (combining different feature types)
4. **Cluster Analysis** (Patent Fig. 5A-C, Fig. 6):
    - Implementation of clustering methods for grouping similar proteins
    - Using DBSCAN for defining protein similarity clusters
    - Similarity threshold adjustment for cluster formation
5. **Data Extraction** (Patent Fig. 8):
    - Feature extraction from structural and sequence data
    - Processing of candidate protein properties
    - Normalization and transformation of features

### Detailed Limitations Compared to Patent

1. **Limited Scope of Implementation**:
    - The patent describes a complete system spanning computational prediction, protein production, and empirical testing (Figure 1: steps 100-800)
    - This implementation covers only steps 200-300 (protein databases and prediction)
    - The system lacks integration with the wet lab components (steps 400-500)
2. **Absence of Protein Production Pipeline** (Patent Fig. 7A):
    - The patent describes detailed protein sourcing methods (native, recombinant, synthesized)
    - Genetic modification of expression hosts
    - Protein purification and formulation steps
    - Chemical modification protocols
    - None of these wet lab procedures are implemented
3. **Limited Feature Sources**:
    - The patent leverages multiple database types (Fig. 2: protein sequence, protein structure, genomic sequence, internal protein database)
    - This implementation primarily uses sequence and predicted structure data
    - Limited integration with specialized databases
4. **Simplified Active Learning**:
    - The patent describes an iterative learning process (Fig. 9) with feedback from empirical testing
    - This implementation has a more basic training pipeline without the complete feedback loop
    - No integration of experimental results back into the model
5. **Single Target Function**:
    - The patent describes a system adaptable to various protein functions (gelation, emulsification, foaming, etc.)
    - This implementation focuses specifically on gelation prediction
    - Model architecture would need adaptation for other target functions