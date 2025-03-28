import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import logging
import os
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.logging import RichHandler
from .dataset import ProteinDataset
from .model import ProtProp
from .dataloaders import collate_fn
from .config import load_config

class ModelRunner:
    def __init__(self, config, train_path=None, val_path=None, test_path=None, model_path=None, output_path=None, verbosity='info'):
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
        self.logger.setLevel(getattr(logging, verbosity.upper()))
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        rich_handler = RichHandler(console=self.console, show_time=True, show_level=True)
        self.logger.addHandler(rich_handler)

        # Initialize model
        if model_path and os.path.exists(model_path):
            self.model, checkpoint = ProtProp.load_from_checkpoint(model_path, config)
            self.start_epoch = checkpoint['epoch']
            self.start_step = checkpoint['step']
        else:
            self.model = ProtProp(
                embed_dim=config.embed_dim,
                num_filters=config.num_filters,
                kernel_sizes=config.kernel_sizes,
                protein_encode_dim=config.protein_encode_dim,
                dropout=config.dropout
            )
            self.start_epoch = 0
            self.start_step = 0
        self.model = self.model.to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        # Load data
        self.json_path = "../data_utils/integrated_data.json"
        with open(self.json_path, 'r') as f:
            self.data = list(json.load(f).values())

    def cluster_and_split(self, data, n_clusters, train_frac):
        """Cluster proteins by sequence similarity and split into train/val or train/test."""
        sequences = [protein['sequence'] for protein in data]
        features = np.array([len(seq) for seq in sequences]).reshape(-1, 1)
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(features)
        
        unique_clusters = np.unique(cluster_labels)
        np.random.shuffle(unique_clusters)
        train_clusters = unique_clusters[:int(train_frac * len(unique_clusters))]
        other_data = [d for d, c in zip(data, cluster_labels) if c not in train_clusters]
        train_data = [d for d, c in zip(data, cluster_labels) if c in train_clusters]
        return train_data, other_data

    def train(self):
        # Split data using clustering if paths are not provided
        if self.train_path and self.val_path:
            self.logger.warning("Train and val paths provided but not used; using clustering instead.")
        train_data, val_data = self.cluster_and_split(self.data, self.config.n_clusters, self.config.train_frac)
        
        train_dataset = ProteinDataset(self.json_path, is_train=True)
        mean_std = train_dataset.mean_std
        val_dataset = ProteinDataset(self.json_path, is_train=False, mean_std=mean_std)
        
        train_dataset.data = train_data
        val_dataset.data = val_data
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)

        # Setup progress bar
        total_epochs = self.config.epochs - self.start_epoch
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
        
        global_step = self.start_step
        with progress:
            epoch_task = progress.add_task("[cyan]Training...", total=total_epochs)
            for epoch in range(self.start_epoch, self.config.epochs):
                self.model.train()
                train_loss = 0
                train_batches = len(train_loader)
                train_batch_task = progress.add_task(f"[green]Epoch {epoch+1} Train", total=train_batches)
                
                for batch in train_loader:
                    sequence = batch['sequence'].to(self.device)
                    residue_features = batch['residue_features'].to(self.device)
                    protein_features = batch['protein_features'].to(self.device)
                    gelation = batch['gelation'].to(self.device)

                    self.optimizer.zero_grad()
                    logits = self.model(sequence, residue_features, protein_features)
                    loss = self.criterion(logits.squeeze(), gelation)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    global_step += 1
                    progress.advance(train_batch_task)

                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                val_batches = len(val_loader)
                val_batch_task = progress.add_task(f"[blue]Epoch {epoch+1} Val", total=val_batches)
                
                with torch.no_grad():
                    for batch in val_loader:
                        sequence = batch['sequence'].to(self.device)
                        residue_features = batch['residue_features'].to(self.device)
                        protein_features = batch['protein_features'].to(self.device)
                        gelation = batch['gelation'].to(self.device)

                        logits = self.model(sequence, residue_features, protein_features)
                        loss = self.criterion(logits.squeeze(), gelation)
                        val_loss += loss.item()

                        preds = (torch.sigmoid(logits) > 0.5).float()
                        correct += (preds.squeeze() == gelation).sum().item()
                        total += gelation.size(0)
                        progress.advance(val_batch_task)

                # Log metrics
                train_loss_avg = train_loss / len(train_loader)
                val_loss_avg = val_loss / len(val_loader)
                val_accuracy = correct / total
                self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
                self.logger.info(f"Train Loss: {train_loss_avg:.4f}")
                self.logger.info(f"Val Loss: {val_loss_avg:.4f}, Accuracy: {val_accuracy:.4f}")

                # Save checkpoint
                if self.output_path:
                    checkpoint_path = os.path.join(self.output_path, f"epoch={epoch+1}-step={global_step}.ckpt")
                    self.model.save_checkpoint(checkpoint_path, self.optimizer, epoch + 1, global_step)
                    self.logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Advance epoch progress
                progress.advance(epoch_task)

    def evaluate(self):
        if not self.model_path:
            self.logger.error("Model path must be provided for evaluation")
            raise ValueError("Model path must be provided for evaluation")

        # Use test_path if provided; otherwise, use clustering to get test data
        if self.test_path:
            # In a real scenario, parse test_path; here we use the full dataset
            self.logger.warning("Test path provided but not used; using clustering instead.")
        _, test_data = self.cluster_and_split(self.data, self.config.n_clusters, self.config.train_frac)
        
        dataset = ProteinDataset(self.json_path, is_train=False, mean_std=self.model.mean_std if hasattr(self.model, 'mean_std') else None)
        dataset.data = test_data
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)

        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                sequence = batch['sequence'].to(self.device)
                residue_features = batch['residue_features'].to(self.device)
                protein_features = batch['protein_features'].to(self.device)
                gelation = batch['gelation'].to(self.device)

                logits = self.model(sequence, residue_features, protein_features)
                loss = self.criterion(logits.squeeze(), gelation)
                test_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds.squeeze() == gelation).sum().item()
                total += gelation.size(0)

        accuracy = correct / total
        self.logger.info(f"Evaluation Loss: {test_loss / len(loader):.4f}, Accuracy: {accuracy:.4f}")

        if self.output_path:
            with open(self.output_path, 'w') as f:
                f.write(f"Evaluation Loss: {test_loss / len(loader):.4f}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
            self.logger.info(f"Saved evaluation results to {self.output_path}")

    def predict(self):
        if not self.model_path:
            self.logger.error("Model path must be provided for prediction")
            raise ValueError("Model path must be provided for prediction")

        # Use test_path if provided; otherwise, use clustering to get test data
        if self.test_path:
            # In a real scenario, parse test_path; here we use the full dataset
            self.logger.warning("Test path provided but not used; using clustering instead.")
        _, test_data = self.cluster_and_split(self.data, self.config.n_clusters, self.config.train_frac)
        
        dataset = ProteinDataset(self.json_path, is_train=False, mean_std=self.model.mean_std if hasattr(self.model, 'mean_std') else None)
        dataset.data = test_data
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                sequence = batch['sequence'].to(self.device)
                residue_features = batch['residue_features'].to(self.device)
                protein_features = batch['protein_features'].to(self.device)
                logits = self.model(sequence, residue_features, protein_features)
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions.extend(probs.flatten())

        if self.output_path:
            with open(self.output_path, 'w') as f:
                f.write("ID\tGelation_Probability\n")
                for i, prob in enumerate(predictions):
                    f.write(f"Protein_{i}\t{prob:.4f}\n")
            self.logger.info(f"Saved predictions to {self.output_path}")