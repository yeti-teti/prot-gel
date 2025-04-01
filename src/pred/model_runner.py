import os
import logging
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.logging import RichHandler

import dask.dataframe as dd
import pyarrow.fs as pafs
from dotenv import load_dotenv

from .dataset import ProteinDataset
from .model import ProtProp
from .dataloaders import collate_fn

# Constants required for predicting
SS_CLASSES = ['H', 'G', 'I', 'E', 'B', 'T', 'S', '-']
AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INT = {aa: i + 1 for i, aa in enumerate(AA_LIST)}
UNKNOWN_AA_INDEX = 0
PADDING_IDX = 0
GELATION_DOMAINS = ["PF00190", "PF04702", "PF00234"]

class ModelRunner:
    def __init__(self, config,
                 r2_main_dataset_path=None, # Path to the COMPLETE dataset on R2
                 train_r2_path=None,
                 val_r2_path=None,
                 test_r2_path=None,
                 mean_std_json_path=None,
                 r2_env_path=".env",
                 model_path=None,
                 output_dir=None,
                 verbosity='info'):
        """
        Initializes the ModelRunner for R2/Parquet data, supporting batch and single prediction.

        Args:
            config: Configuration object... (same as before)
            r2_main_dataset_path (str): R2 path (relative to bucket) to the COMPLETE Parquet dataset directory (used for predict_single).
            train_r2_path (str, optional): R2 path for training data...
            val_r2_path (str, optional): R2 path for validation data...
            test_r2_path (str, optional): R2 path for test data...
            mean_std_json_path (str): Local path to the mean_std.json file. REQUIRED.
            r2_env_path (str): Path to the .env file with R2 credentials...
            model_path (str, optional): Path to a saved checkpoint file (.ckpt)...
            output_dir (str, optional): Local directory to save checkpoints and output files...
            verbosity (str, optional): Logging level...
        """
        self.config = config
        # Store all paths
        self.r2_main_dataset_path = r2_main_dataset_path
        self.train_r2_path = train_r2_path
        self.val_r2_path = val_r2_path
        self.test_r2_path = test_r2_path
        self.mean_std_json_path = mean_std_json_path
        self.r2_env_path = r2_env_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.console = Console()
        self._setup_logging(verbosity) # Logging setup helper
        self.logger.info(f"Using device: [bold blue]{self.device}[/]")

        # --- Load mean_std statistics (Required for all operations) ---
        self.mean_std = self._load_mean_std_from_file()

        # --- Initialize model structure ---
        self.logger.info("Initializing model architecture...")
        self.model = ProtProp(
            embed_dim=getattr(config, 'embed_dim', 64), # Use getattr for safety
            num_filters=getattr(config, 'num_filters', 64),
            kernel_sizes=getattr(config, 'kernel_sizes', [3,5,7]),
            protein_encode_dim=getattr(config, 'protein_encode_dim', 32),
            dropout=getattr(config, 'dropout', 0.4)
        ).to(self.device)
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # --- Define Optimizer (Needed for training and potentially loading state) ---
        self.optimizer = optim.Adam(self.model.parameters(), lr=getattr(config, 'lr', 1e-4))
        self.logger.info(f"Optimizer: Adam, LR: {getattr(config, 'lr', 1e-4)}")

        # --- Intialize training state variables ---
        self.start_epoch = 0
        self.global_step = 0

        # --- Load Checkpoint (if specified) ---
        if self.model_path:
            self._load_checkpoint() # Checkpoint loading helper
        else:
            self.logger.info("No model checkpoint specified. Model initialized with random weights.")

        # --- Loss function (Needed for training/evaluation) ---
        self.criterion = nn.BCEWithLogitsLoss()

        # --- Setup Output Directory ---
        self._setup_output_dir() # Output dir helper

        # --- R2 Filesystem (Needed for predict_single) ---
        self._r2_fs = None # Lazy initialize filesystem for single prediction
        self._ddf_main = None # Lazy initialize main ddf for single prediction
    
    def _setup_logging(self, verbosity):
        """Sets up Rich logging."""
        self.logger = logging.getLogger("model_runner")
        self.logger.handlers.clear()
        log_level = getattr(logging, verbosity.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        rich_handler = RichHandler(console=self.console, show_time=False, show_level=True, markup=True, log_time_format="[%X]")
        formatter = logging.Formatter('%(message)s')
        rich_handler.setFormatter(formatter)
        self.logger.addHandler(rich_handler)
        self.logger.propagate = False

    def _load_mean_std_from_file(self):
        """Loads and validates mean_std stats from the JSON file."""
        if not self.mean_std_json_path or not os.path.exists(self.mean_std_json_path):
            self.logger.error(f"mean_std.json file not found or path not provided: {self.mean_std_json_path}")
            raise FileNotFoundError(f"mean_std.json is required.")
        try:
            with open(self.mean_std_json_path, 'r') as f: stats_data = json.load(f)
            mean_std_dict = {k: np.array(v, dtype=np.float64) for k, v in stats_data.items() if k in ['protein_mean', 'protein_std', 'residue_mean', 'residue_std']}
            if len(mean_std_dict) != 4: raise KeyError("Missing required keys in mean_std file.")
            return mean_std_dict
        except Exception as e:
            self.logger.error(f"Failed to load or process {self.mean_std_json_path}: {e}", exc_info=True)
            raise
    
    def _load_checkpoint(self):
        """Loads model and optimizer state from self.model_path."""
        if not os.path.exists(self.model_path):
            self.logger.error(f"Checkpoint path specified but not found: {self.model_path}")
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")

        self.logger.info(f"Attempting to load checkpoint from: [green]{self.model_path}[/]")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Load optimizer state only if it exists in the checkpoint
            if 'optimizer_state_dict' in checkpoint:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 self.logger.info("Loaded optimizer state from checkpoint.")
            else:
                 self.logger.warning("Optimizer state not found in checkpoint. Using fresh optimizer state.")
            self.start_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('step', 0)

            self.logger.info(f"Model loaded. Resuming from Epoch {self.start_epoch + 1}, Step {self.global_step}")

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}. Model weights might be random.", exc_info=True)
            # Reset state variables if loading fails, keep random weights in model
            self.start_epoch = 0
            self.global_step = 0
            self.optimizer = optim.Adam(self.model.parameters(), lr=getattr(self.config, 'lr', 1e-4))

    def _setup_output_dir(self):
        """Creates the output directory if specified and valid."""
        if self.output_dir:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                self.logger.info(f"Output directory set to: [green]{self.output_dir}[/]")
            except OSError as e:
                 self.logger.error(f"Failed to create output directory {self.output_dir}: {e}. Checkpoint saving/output disabled.")
                 self.output_dir = None
        else:
            self.logger.warning("No output directory specified. Checkpoint saving and result output will be disabled.")

    def _save_checkpoint(self, epoch, step):
        """Saves the current model, optimizer, state and mean_std"""
        if not self.output_dir: return
        checkpoint_name = f"epoch={epoch}-step={step}.ckpt"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        try:
            save_data = {
                'epoch': epoch, 'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'mean_std': {k: v.tolist() for k, v in self.mean_std.items()},
                'config': vars(self.config) if hasattr(self.config, '__dict__') else str(self.config)
            }
            torch.save(save_data, checkpoint_path)
            self.logger.debug(f"Checkpoint saved to: [green]{checkpoint_path}[/]")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}", exc_info=False)

    def _get_r2_fs(self):
        """Initializes and returns the R2 filesystem object."""
        if self._r2_fs is None:
            self.logger.debug("Initializing R2 filesystem connection...")
            try:
                 load_dotenv(dotenv_path=self.r2_env_path)
                 access_key = os.getenv("CLOUDFARE_ACCESS_KEY")
                 secret_key = os.getenv("CLOUDFARE_SECRET_KEY")
                 account_id = os.getenv("CLOUDFARE_ACCOUNT_ID")
                 endpoint = os.getenv("CLOUDFARE_ENDPOINT")
                 if not endpoint and account_id: endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
                 if not all([access_key, secret_key, endpoint]): raise ValueError("Missing R2 credentials")
                 self._r2_fs = pafs.S3FileSystem(endpoint_override=endpoint, access_key=access_key, secret_key=secret_key, scheme="https")
                 self.logger.debug("R2 filesystem initialized.")
            except Exception as e:
                 self.logger.error(f"Failed to initialize R2 filesystem from {self.r2_env_path}: {e}")
                 raise
        return self._r2_fs
    
    def _get_main_ddf(self):
        """Initializes and returns the Dask DataFrame for the main dataset."""
        if self._ddf_main is None:
            if not self.r2_main_dataset_path:
                    raise ValueError("r2_main_dataset_path is required for single prediction but not set.")
            self.logger.info(f"Initializing Dask DataFrame for main dataset: [green]{self.r2_main_dataset_path}[/]")
            try:
                # Need R2 bucket name
                load_dotenv(dotenv_path=self.r2_env_path)
                r2_bucket_name = os.getenv("CLOUDFARE_BUCKET_NAME")
                if not r2_bucket_name: 
                    raise ValueError("R2 bucket name not found in .env")

                storage_options = self._get_r2_fs().storage_options # Get storage options from initialized fs
                full_uri = f"r2://{r2_bucket_name}/{self.r2_main_dataset_path}"
                # Define columns needed for processing a single protein
                columns_needed = [
                        'uniprot_id', 'sequence', 'sequence_length',
                        'residue_features', 'structural_features',
                        'physicochemical_properties', 'domains', 'gelation' # Gelation might not be needed for prediction
                ]
                self._ddf_main = dd.read_parquet(full_uri, columns=columns_needed, storage_options=storage_options)
                self.logger.info(f"Main Dask DataFrame initialized ({self._ddf_main.npartitions} partitions).")
            except Exception as e:
                self.logger.error(f"Failed to initialize main Dask DataFrame from {full_uri}: {e}", exc_info=True)
                raise
        return self._ddf_main

    def _get_dataloader(self, r2_data_path, is_train):
        """Helper to create dataset and dataloader"""
        if not r2_data_path:
            self.logger.error(f"R2 data path is missing for {'training' if is_train else 'validation/test'}.")
            raise ValueError("R2 data path not provided")

        try:
            # Need bucket name from env for dataset init
            load_dotenv(dotenv_path=self.r2_env_path)
            r2_bucket_name = os.getenv("CLOUDFARE_BUCKET_NAME")
            if not r2_bucket_name: raise ValueError("R2 bucket name not found in .env")

            dataset = ProteinDataset(
                r2_dataset_path=r2_data_path,
                mean_std_path=self.mean_std_json_path,
                r2_config_env_path=self.r2_env_path,
                r2_bucket_name=r2_bucket_name # Pass bucket name
            )
            if len(dataset) == 0:
                self.logger.warning(f"Loaded dataset from {r2_data_path} but it has length 0.")
                return None

            loader = DataLoader(
                dataset, batch_size=self.config.batch_size,
                shuffle=is_train, collate_fn=collate_fn,
                num_workers=getattr(self.config, 'num_workers', 0),
                pin_memory=True if self.device.type == 'cuda' else False
            )
            return loader
        except Exception as e:
            self.logger.error(f"Failed to create DataLoader for R2 path {r2_data_path}: {e}", exc_info=True)
            raise

    def train(self):
        """
            Training the model with the train and val data.
        """
        if not self.train_r2_path or not self.val_r2_path:
            self.logger.error("R2 paths for train and validation data must be provided for training.")
            raise ValueError("Training requires train_r2_path and val_r2_path.")

        self.logger.info(f"Starting training process...")
        self.logger.info(f"  Training data R2 path: [green]{self.train_r2_path}[/]")
        self.logger.info(f"  Validation data R2 path: [green]{self.val_r2_path}[/]")
        self.logger.info(f"  Mean/Std stats: [green]{self.mean_std_json_path}[/]")
        self.logger.info(f"  Epochs: {self.config.epochs}, Start Epoch: {self.start_epoch + 1}")
        self.logger.info(f"  Batch Size: {self.config.batch_size}")
        self.logger.info(f"  Learning Rate: {self.config.lr}")

        # Get DataLoaders using the helper
        train_loader = self._get_dataloader(self.train_r2_path, is_train=True)
        val_loader = self._get_dataloader(self.val_r2_path, is_train=False)

        if train_loader is None or val_loader is None:
             self.logger.error("Failed to create one or both DataLoaders. Aborting training.")
             return

        self.logger.info(f"Training set size: {len(train_loader.dataset):,}, Validation set size: {len(val_loader.dataset):,}")

        # Training loop state
        # global_step already initialized from checkpoint or 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # Progress bar setup
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn(" [blue]Loss:[/] {task.fields[loss]:.4f}"),
            TextColumn("[blue]Acc:[/] {task.fields[acc]:.3f}"),# If accuracy is relevant
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )

        with progress:
            epoch_task = progress.add_task("[cyan]Epochs", total=self.config.epochs, loss=float('nan'), acc=float('nan'))
            progress.update(epoch_task, completed=self.start_epoch)

            for epoch in range(self.start_epoch, self.config.epochs):
                epoch_desc = f"[cyan]Epoch {epoch+1}/{self.config.epochs}"
                progress.update(epoch_task, description=epoch_desc, loss=float('nan'), acc=float('nan')) # Reset epoch stats display

                # --- Training phase ---
                self.model.train()
                train_loss_accum = 0.0
                train_batches = len(train_loader)
                train_batch_task = progress.add_task(f"[green]  Train", total=train_batches, loss=float('nan'), acc=float('nan')) # No acc needed here

                for batch_idx, batch in enumerate(train_loader):
                    if batch is None: # Skip batches that failed collation
                        self.logger.warning(f"Skipping None batch at epoch {epoch+1}, train batch index {batch_idx}")
                        progress.update(train_batch_task, advance=1)
                        continue

                    try:
                        sequence = batch['sequence'].to(self.device, non_blocking=True)
                        residue_features = batch['residue_features'].to(self.device, non_blocking=True)
                        protein_features = batch['protein_features'].to(self.device, non_blocking=True)
                        gelation = batch['gelation'].to(self.device, non_blocking=True) # [batch, 1]

                        self.optimizer.zero_grad()
                        logits = self.model(sequence, residue_features, protein_features) # [batch, 1]
                        loss = self.criterion(logits, gelation) # BCEWithLogits expects [batch, 1] for both

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step() # Corrected typo

                        current_loss = loss.item()
                        train_loss_accum += current_loss
                        self.global_step += 1

                        # Update progress bar for batch
                        progress.update(train_batch_task, advance=1, description=f"[green]  Train", loss=current_loss)

                        # Log periodically
                        if self.global_step % self.config.log_interval == 0:
                            self.logger.debug(f"E{epoch+1} B{batch_idx+1}/{train_batches} | Step {self.global_step} | Loss: {current_loss:.4f}")

                    except Exception as e:
                         self.logger.error(f"Error during training batch {batch_idx} in epoch {epoch+1}: {e}", exc_info=True)
                         raise # Stop training

                progress.remove_task(train_batch_task) # Clean up task for the epoch

                # --- Validation phase ---
                self.model.eval()
                val_loss_accum = 0.0
                correct = 0
                total = 0
                val_batches = len(val_loader)
                all_val_labels = []
                all_val_preds = []
                val_batch_task = progress.add_task(f"[blue]   Validate", total=val_batches, loss=float('nan'), acc=float('nan'))

                with torch.no_grad():
                    for batch in val_loader:
                        if batch is None:
                            self.logger.warning(f"Skipping None batch during validation epoch {epoch+1}")
                            progress.update(val_batch_task, advance=1)
                            continue

                        try:
                            sequence = batch['sequence'].to(self.device, non_blocking=True)
                            residue_features = batch['residue_features'].to(self.device, non_blocking=True)
                            protein_features = batch['protein_features'].to(self.device, non_blocking=True)
                            gelation = batch['gelation'].to(self.device, non_blocking=True) # [batch, 1]

                            logits = self.model(sequence, residue_features, protein_features) # [batch, 1]
                            loss = self.criterion(logits, gelation)
                            val_loss_accum += loss.item()

                            probabilities = torch.sigmoid(logits) # [batch, 1]
                            preds = (probabilities > 0.5).float() # [batch, 1]
                            correct += (preds == gelation).sum().item()
                            total += gelation.size(0)

                            all_val_preds.extend(probabilities.cpu().numpy().flatten())
                            all_val_labels.extend(gelation.cpu().numpy().flatten())

                            progress.update(val_batch_task, advance=1, description=f"[blue]   Validate")

                        except Exception as e:
                            self.logger.error(f"Error during validation batch in epoch {epoch+1}: {e}", exc_info=True)
                            # Continue validation if possible
                            progress.update(val_batch_task, advance=1)


                progress.remove_task(val_batch_task) # Clean up task for the epoch

                # Calculate epoch metrics
                avg_train_loss = train_loss_accum / train_batches if train_batches > 0 else 0
                avg_val_loss = val_loss_accum / val_batches if val_batches > 0 else float('inf')
                val_accuracy = correct / total if total > 0 else 0
                val_auc = roc_auc_score(all_val_labels, all_val_preds) if len(np.unique(all_val_labels)) > 1 else float('nan')

                # Update overall epoch progress bar display
                progress.update(epoch_task, advance=1, loss=avg_val_loss, acc=val_accuracy)

                # Log epoch summary
                self.logger.info(f"[bold]Epoch {epoch+1}/{self.config.epochs} Summary:[/]")
                self.logger.info(f"  Train Loss: {avg_train_loss:.4f}")
                self.logger.info(f"  Val Loss  : {avg_val_loss:.4f}")
                self.logger.info(f"  Val Acc   : {val_accuracy:.4f} ({correct}/{total})")
                if not np.isnan(val_auc):
                    self.logger.info(f"  Val AUC   : {val_auc:.4f}")


                # Checkpointing and Early Stopping Logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.logger.info(f"  New best validation loss: {best_val_loss:.4f}. Saving checkpoint...")
                    self._save_checkpoint(epoch + 1, self.global_step) # Save after completing epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    self.logger.info(f"  Validation loss did not improve ({avg_val_loss:.4f} vs best {best_val_loss:.4f}). {epochs_without_improvement} epochs without improvement.")
                    if getattr(self.config, 'save_every_epoch', False):
                         self._save_checkpoint(epoch + 1, self.global_step)

                # Early stopping condition
                early_stopping_patience = getattr(self.config, 'early_stopping_patience', None)
                if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                    self.logger.warning(f"Stopping early after {epochs_without_improvement} epochs without validation loss improvement.")
                    break # Exit epoch loop

            # End of training loop
            progress.update(epoch_task, description="[cyan]Epochs Complete")

        self.logger.info("Training finished.")


    def evaluate(self, output_file="evaluation_results.txt"):
        """Evalautes on the test set"""

        if self.global_step == 0 and not (self.model_path and os.path.exists(self.model_path)):
            self.logger.error("No trained model available. Load checkpoint or train first.")
            raise RuntimeError("Model not loaded/trained.")
        if not self.test_r2_path: 
            raise ValueError("test_r2_path required.")
        if self.mean_std is None: 
            raise ValueError("mean_std required.")

        self.logger.info(f"Starting evaluation on test data (R2 Path): [green]{self.test_path}[/]")

        # Get Dataloader
        test_loader = self._get_dataloader(self.test_r2_path, is_train=False)
        if test_loader is None:
            self.logger.error("Failed to create test DataLoader.")
            return
        self.logger.info(f"Test set size: {len(test_loader.dataset):,}")

        self.model.eval()
        test_loss_accum = 0.0
        correct = 0
        total = 0
        all_preds_probs = []
        all_labels = []
        all_ids = []

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("Loss: {task.fields[loss]:.4f}"),
            TextColumn("Acc: {task.fields[acc]:.3f}"),
            TimeElapsedColumn(),
            console=self.console
        )

        with progress:
            eval_task = progress.add_task("[magenta]Evaluating", total=len(test_loader), loss=float('nan'), acc=float('nan'))
            
            with torch.no_grad():
                for batch in test_loader:
                    if batch is None:
                        self.logger.warning("Skipping None batch during evaluation")
                        progress.update(eval_task, advance=1)
                        continue
                    
                    try:
                        sequence = batch['sequence'].to(self.device, non_blocking=True)
                        residue_features = batch['residue_features'].to(self.device, non_blocking=True)
                        protein_features = batch['protein_features'].to(self.device, non_blocking=True)
                        gelation = batch['gelation'].to(self.device, non_blocking=True) # Ground truth labels [batch, 1]
                        ids_batch = batch['uniprot_ids'] # List of IDs

                        logits = self.model(sequence, residue_features, protein_features) # [batch, 1]
                        loss = self.criterion(logits, gelation)
                        test_loss_accum += loss.item()

                        probabilities = torch.sigmoid(logits) # [batch, 1]
                        preds_binary = (probabilities > 0.5).float() # [batch, 1]
                        correct += (preds_binary == gelation).sum().item()
                        total += gelation.size(0)

                        all_preds_probs.extend(probabilities.cpu().numpy().flatten())
                        all_labels.extend(gelation.cpu().numpy().flatten())
                        all_ids.extend(ids_batch)

                        # Update progress for batch
                        batch_acc = (preds_binary == gelation).sum().item() / gelation.size(0) if gelation.size(0) else 0
                        progress.update(eval_task, advance=1, loss=loss.item(), acc=batch_acc)
                    except Exception as e:
                        self.logger.error(f"Error during evaluation batch: {e}", exc_info=True)
                        progress.update(eval_task, advance=1)
        
        # Calculate oveerall metrics
        final_accuracy = correct / total if total > 0 else 0
        avg_test_loss = test_loss_accum / len(test_loader) if len(test_loader) > 0 else 0
        self.logger.info(f"Evaluation finished.")
        self.logger.info(f"  Test Loss    : {avg_test_loss:.4f}")
        self.logger.info(f"  Test Accuracy: {final_accuracy:.4f} ({correct}/{total})")

        auc = float('nan')
        if len(np.unique(all_labels)) > 1:
            try:
                auc = roc_auc_score(all_labels, all_preds_probs)
                self.logger.info(f"  Test AUC     : {auc:.4f}")
            except ValueError as e:
                 self.logger.warning(f"Could not calculate AUC: {e}")
        else:
            self.logger.warning("Skipping AUC calculation as only one class present in test labels.")

        # Save evaluation metrics
        if self.output_dir:
            output_filepath = os.path.join(self.output_dir, output_file)
            try:
                with open(output_filepath, 'w') as f:
                    f.write(f"Evaluation Results for model: {self.model_path or 'Trained in memory'}\n")
                    f.write(f"Test data R2 path: {self.test_r2_path}\n")
                    f.write(f"Test Loss: {avg_test_loss:.4f}\n")
                    f.write(f"Accuracy: {final_accuracy:.4f} ({correct}/{total})\n")
                    if not np.isnan(auc):
                        f.write(f"AUC: {auc:.4f}\n")
                self.logger.info(f"Saved evaluation results to: [green]{output_filepath}[/]")
            except Exception as e:
                self.logger.error(f"Failed to save evaluation results to {output_filepath}: {e}", exc_info=True)

        self.logger.info("Evaluation finished.")

    def predict(self, predict_r2_path=None, out_file="predictions.tsv"):
        """Generates predictions for data in input path"""

        if self.global_step == 0 and not (self.model_path and os.path.exists(self.model_path)):
             self.logger.error("No trained model available. Load checkpoint or train first.")
             raise RuntimeError("Model not loaded/trained.")
        pred_input_r2_path = predict_r2_path or self.test_r2_path
        if not pred_input_r2_path: 
            raise ValueError("R2 data path required.")
        if self.mean_std is None: 
            raise ValueError("mean_std required.")

        self.logger.info(f"Starting batch prediction on data (R2 path): [green]{pred_input_r2_path}[/]")
        pred_loader = self._get_dataloader(pred_input_r2_path, is_train=False)
        if pred_loader is None:
            return None, None

        self.logger.info(f"Prediction set size: {len(pred_loader.dataset):, }")

        self.model.eval()
        all_predictions_probs = []
        all_ids = []

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console
        )

        with progress:
            pred_task = progress.add_task("[yellow]Predicting", total=len(pred_loader))
            with torch.no_grad():
                for batch in pred_loader:
                    if batch is None:
                        self.logger.warning("Skipping None batch during prediction.")
                        progress.update(pred_task, advance=1)
                        continue
                    
                    try:
                        sequence = batch['sequence'].to(self.device, non_blocking=True)
                        residue_features = batch['residue_features'].to(self.device, non_blocking=True)
                        protein_features = batch['proterin_features'].to(self.device, non_blocking=True)
                        ids_batch = batch['uniprot_ids']

                        logits = self.model(sequence, residue_features, protein_features) # [batch, 1]
                        probs = torch.sigmoid(logits) # [batch, 1]

                        all_predictions_probs.extend(probs.cpu().numpy().flatten())
                        all_ids.extend(ids_batch)
                        progress.update(pred_task, advance=1)

                    except Exception as e:
                        self.logger.error(f"Error during prediciotn batch: {e}", exc_info=True)
                        progress.update(pred_task, advance=1)

        self.logger.info("Predictions finished.")

        # Check consistency
        if len(all_ids) != len(all_predictions_probs):
            self.logger.error(f"Mismatch between number of IDs ({len(all_ids)}) and predictions ({len(all_predictions_probs)}). Output may be incorrect.")

            return None

        if self.output_dir:
            output_filepath = os.path.join(self.output_dir, out_file)
            try:
                with open(output_filepath, 'w') as f:
                    f.write("ID\tGelation_Probability\n")
                    for prot_id, prob in zip(all_ids, all_predictions_probs):
                        f.write(f"{prot_id}\t{prob:.6f}\n") # Use more precision for probabilities
                self.logger.info(f"Saved predictions to: [green]{output_filepath}[/]")
            except Exception as e:
                self.logger.error(f"Failed to save predictions to {output_filepath}: {e}", exc_info=True)

        # Return predictions and IDs
        return all_ids, all_predictions_probs