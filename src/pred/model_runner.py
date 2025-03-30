import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import json
import numpy as np
from sklearn.metrics import roc_auc_score
import logging
import os
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.logging import RichHandler

from .dataset import ProteinDataset
from .model import ProtProp
from .dataloaders import collate_fn

class ModelRunner:
    def __init__(self, config, train_path=None, val_path=None, test_path=None, model_path=None, output_path=None, verbosity='info'):
        """
        Initializes the ModelRunner.

        Args:
            config: Configuration object with model and training parameters.
            train_path (str, optional): Path to the training data JSON file. Required for training.
            val_path (str, optional): Path to the validation data JSON file. Required for training.
            test_path (str, optional): Path to the test data JSON file. Required for evaluation/prediction.
            model_path (str, optional): Path to a saved checkpoint file to load model/optimizer/state from.
            output_dir (str, optional): Directory to save checkpoints and output files.
            verbosity (str, optional): Logging level ('debug', 'info', 'warning', 'error'). Defaults to 'info'.
        """
        self.config = config
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model_path = model_path
        self.output_path = output_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.console = Console()
        
        # Setup logging
        self.logger = logging.getLogger("model_runner")
        self.logger.handlers.clear()
        self.logger.setLevel(getattr(logging, verbosity.upper()))
        # Rich Hnadler
        rich_handler = RichHandler(console=self.console, show_time=True, show_level=True, markup=True)
        formatter = logging.Formatter('%(message)s') 
        rich_handler.setFormatter(formatter)
        self.logger.addHandler(rich_handler)
        self.logger.propagate = False 
        self.logger.info(f"Using device: [bold blue]{self.device}[/]")

        # Initialize model structure
        self.model = ProtProp(
            embed_dim=self.config.embed_dim,
            num_filters=self.config.num_filters,
            kernel_sizes=self.config.kernel_sizes,
            protein_encode_dim=self.config.protein_encode_dim,
            dropout=self.config.dropout
        )

        # Intialize training state variables
        self.start_epoch = 0
        self.start_step = 0
        self.mean_std = None

        # If checkpoint provided
        if self.model_path and os.path.exists(model_path):
            self.logger.info(f"Loading checkpoint from: [green]{self.model_path}[/]")
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimzier_state_dict'])
                self.start_epoch = checkpoint['epoch']
                self.start_step = checkpoint.get('step', 0)
                self.mean_std = checkpoint.get('mean_std')
                self.logger.info(f"Resuming training from epoch {self.start_epoch}, step {self.start_step}")
                if self.mean_std:
                    self.logger.info("Loaded mean_std from checkpoint")
                else:
                     self.logger.warning("mean_std not found in checkpoint. Must be calculated during training.")
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {e}. Starting from scratch.", exc_info=True)
                self.start_epoch = 0
                self.start_step = 0
                self.mean_std = None
        else:
            self.logger.info("No checkpoint. Starting from scratch")
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Output Directory
        if self.output_path and not os.path.exists(output_path):
            os.makedirs(self.output_path)
            self.logger.info(f"Created output directory: [green]{self.output_path}[/]")
    
    def _save_checkpoint(self, epoch, step):
        """Saves the current model, optimizer, state and mean_std"""
        if not self.output_path:
            self.logger.warning("Output directory not set. Skipping checkpoint saving.")
            return
        
        checkpoint_path = os.path.join(self.output_dir, f"epoch={epoch}-step={step}.ckpt")
        save_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'mean_std': self.mean_std, # Save the normalization stats
            'config': self.config # Optionally save config used for this run
        }

    def train(self):
        """
            Training the model with the train and val data.
        """
        if not self.train_path or not self.val_path:
            raise ValueError("Path to train and val data not provided")
        if not os.path.exists(self.train_path):
            raise ValueError("Training data not found")
        if not os.path.exists(self.val_path):
            raise ValueError("Val data not found")
        
        self.logger.info(f"Starting training process...")
        self.logger.info(f"  Training data: [green]{self.train_path}[/]")
        self.logger.info(f"  Validation data: [green]{self.val_path}[/]")
        self.logger.info(f"  Epochs: {self.config.epochs}, Start Epoch: {self.start_epoch}")
        self.logger.info(f"  Batch Size: {self.config.batch_size}")
        self.logger.info(f"  Learning Rate: {self.config.lr}")

        # Dataset and Dataloader
        # Calculate the mean_std from training data
        if self.mean_std is None:
            self.logger.info("Calculating mean_std from training data...")
            try:
                temp_train_dataset = ProteinDataset(
                    json_path=self.train_path,
                    is_train=True
                )
                self.mean_std = temp_train_dataset.mean_std
                self.logger.info("mean_std computed.")
                del temp_train_dataset
            except Exception as e:
                self.logger.error(f"Failed to calcualte mean_std")
                raise
        
        # Create dataset
        try:
            train_dataset = ProteinDataset(json_path=self.train_path, is_train=True, mean_std=self.mean_std)
            val_dataset = ProteinDataset(json_path=self.val_path, is_train=False, mean_std=self.mean_std)
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}", exc_info=True)
            raise

        train_loader = DataLoader(
            train_dataset, 
            batch_size = self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.get('num_workers', 0)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size = self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.get('num_workers', 0)
        )
        self.logger.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

        # Training loop
        global_step = self.start_step
        best_val_loss = float('inf')

        # Progress bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
            console=self.console
        )

        with progress:
            epoch_task = progress.add_task("[cyan]Epochs", total=self.config.epochs)
            progress.update(epoch_task, completed=self.start_epoch)

            for epoch in range(self.start_epoch, self.config.epochs):
                epoch_desc = f"[cyan]Epoch {epoch+1}/{self.config.epochs}"
                progress.update(epoch_task, description=epoch_desc)

                # Training phase
                self.model.train()
                train_loss = 0
                train_batches = len(train_loader)
                train_batch_task = progress.add_task(f"[green] Train, total=train_batches")

                for batch_idx, batch in enumerate(train_loader):
                    sequence = batch['sequence'].to(self.device)
                    residue_features = batch['residue_features'].to(self.device)
                    protein_features = batch['protein_features'].to(self.device)
                    gelation = batch['gelation'].to(self.device)

                    self.optimizer.zero_grad()
                    logits = self.model(sequence, residue_features, protein_features)
                    loss = self.criterion(logits.squeeze(1), gelation)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1.0)
                    self.optimzier.step()
                    train_loss += loss.item()
                    global_step += 1
                    progress.update(train_batch_task, advance=1, description=f"[green]  Train Loss: {loss.item():.4f}")

                    if batch_idx % self.config.log_interval == 0:
                        self.logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}/{train_batches}, Train Loss: {loss.item():.4f}")
                progress.remove_task(train_batch_task) # Clean up task for the epoch

                # Validation phase
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                val_batches = len(val_loader)
                val_batch_task = progress.add_task(f"[blue]  Validate", total=val_batches)

                with torch.no_grad():
                    for batch in val_loader:
                        sequence = batch['sequence'].to(self.device)
                        residue_features = batch['residue_features'].to(self.device)
                        protein_features = batch['protein_features'].to(self.device)
                        gelation = batch['gelation'].to(self.device)

                        logits = self.model(sequence, residue_features, protein_features)
                        loss = self.criterion(logits.squeeze(1), gelation)
                        val_loss += loss.item()

                        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
                        correct += (preds == gelation).sum().item()
                        total += gelation.size(0)
                        progress.update(val_batch_task, advance=1, description=f"[blue]  Validate")
                progress.remove_task(val_batch_task) # Clean up task for the epoch

                # Log for the epoch
                train_loss_avg = train_loss / train_batches if train_batches > 0 else 0
                val_loss_avg = val_loss / val_batches if val_batches > 0 else 0
                val_accuracy = correct / total if total > 0 else 0
                self.logger.info(f"Epoch {epoch+1}/{self.config.epochs} Summary:")
                self.logger.info(f"  Train Loss: {train_loss_avg:.4f}")
                self.logger.info(f"  Val Loss: {val_loss_avg:.4f}, Val Accuracy: {val_accuracy:.4f}")
                
                # Save checkpoints
                if val_loss_avg < best_val_loss:
                     best_val_loss = val_loss_avg
                     self.logger.info(f"  New best validation loss: {best_val_loss:.4f}. Saving checkpoint...")
                     self._save_checkpoint(epoch + 1, global_step) # Save after completing epoch
                elif self.config.get('save_every_epoch', False):
                     self._save_checkpoint(epoch + 1, global_step)

                progress.update(epoch_task, advance=1) # Advance overall epoch progress
        self.logger.info("Training finished.")

    def evaluate(self, output_file="evaluation_results.txt"):
        """Evalautes on the test set"""
        if not self.model_path and self.start_epoch == 0:
            self.logger.error("No trained model available. Load a checkpoint or train first.")
            raise RuntimeError("Model has not been loaded or trained.")
        if not self.test_path:
             raise ValueError("test_path must be provided for evaluation.")
        if not os.path.exists(self.test_path):
             raise FileNotFoundError(f"Test data not found at: {self.test_path}")
        if self.mean_std is None:
            # Attempt to load from model_path again if not present
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.mean_std = checkpoint.get('mean_std')
            if self.mean_std is None:
                raise ValueError("mean_std not available. Load from checkpoint with mean_std or train first.")
            else:
                self.logger.info("Loaded mean_std from checkpoint for evaluation.")

        self.logger.info(f"Starting evaluation on test data: [green]{self.test_path}[/]")

        try:
            test_dataset = ProteinDataset(json_path=self.test_path, is_train=False, mean_std=self.mean_std)
        except Exception as e:
            self.logger.error(f"Failed to load test dataset: {e}", exc_info=True)
            raise

        loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.config.get('num_workers', 0))
        self.logger.info(f"Test set size: {len(test_dataset)}")

        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        progress = Progress(
            TextColumn["[progress.description]{task.description}"],
            BarColumn(),
            MofNCompleteColumn(),
            console = self.console
        )

        with progress:
            eval_task = progress.add_task("[magenta]Evaluating", total=len(loader))
            with torch.no_grad():
                for batch in loader:
                    sequence = batch['sequence'].to(self.device)
                    residue_features = batch['residue_features'].to(self.device)
                    protein_features = batch['protein_features'].to(self.device)
                    gelation = batch['gelation'].to(self.device) # Ground truth labels

                    logits = self.model(sequence, residue_features, protein_features)
                    loss = self.criterion(logits.squeeze(1), gelation)
                    test_loss += loss.item()

                    probabilities = torch.sigmoid(logits.squeeze(1))
                    preds = (probabilities > 0.5).float()
                    correct += (preds == gelation).sum().item()
                    total += gelation.size(0)

                    all_preds.extend(probabilities.cpu().numpy())
                    all_labels.extend(gelation.cpu().numpy())
                    progress.update(eval_task, advance=1)
        
            accuracy = correct / total if total > 0 else 0
            avg_test_loss = test_loss / len(loader) if len(loader) > 0 else 0
            self.logger.info(f"Evaluation finished.")
            self.logger.info(f"  Test Loss: {avg_test_loss:.4f}")
            self.logger.info(f"  Test Accuracy: {accuracy:.4f} ({correct}/{total})")

            if len(np.unique(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_preds)
                self.logger.info(f" Test AUC: {auc:.4f}")
            else:
                self.logger.warning("Skipping AUC calculation as only one class present in test labels.")
            
            # Evalaution metrics
            if self.output_path:
                output_dir = os.path.join(self.output_path, output_file)
                try:
                    with open(output_dir, 'w') as f:
                        f.write(f"Evaluation Results for model: {self.model_path or 'Trained in memory'}\n")
                        f.write(f"Test data: {self.test_path}\n")
                        f.write(f"Test Loss: {avg_test_loss:.4f}\n")
                        f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")
                        if len(np.unique(all_labels)) > 1:
                            f.write(f"AUC: {auc:.4f}\n")
                    self.logger.info(f"Saved evaluation results to: [green]{output_dir}[/]")
                except Exception as e:
                    self.logger.error(f"Failed to save evaluation results to {output_dir}: {e}", exc_info=True)

    def predict(self, input_path=None, out_file="predictions.tsv"):
        """Generates predictions for data in input path"""
        if not self.model_path and self.start_epoch == 0:
            self.logger.error("No trained model available. Load a checkpoint or train first.")
            raise RuntimeError("Model has not been loaded or trained.")
        
        pred_input_path = input_path or self.test_path
        if not pred_input_path:
            raise ValueError("Input data path must be provided (either via input_path or test_path).")
        if not os.path.exists(pred_input_path):
            raise FileNotFoundError(f"Prediction input data not found at: {pred_input_path}")
        if self.mean_std is None:
            # Attempt to load from model_path again if not present
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.mean_std = checkpoint.get('mean_std')
            if self.mean_std is None:
                raise ValueError("mean_std not available. Load from checkpoint with mean_std or train first.")
            else:
                self.logger.info("Loaded mean_std from checkpoint for prediction.")

        self.logger.info(f"Starting prediction on data: [green]{pred_input_path}[/]")

        # Load original data to get IDs
        
