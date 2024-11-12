import pytorch_lightning as pl
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast
import torch.nn as nn
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import Callback
import os
import matplotlib.pyplot as plt


def _convert_to_dataset(data):
    """Convert raw data to PyTorch Dataset."""
    if isinstance (data, np.ndarray):
        data = torch.from_numpy (data)

    if isinstance (data, torch.Tensor):
        # Assuming last column is target for structured data
        if len (data.shape) == 2:
            features = data [:, :-1]
            targets = data [:, -1]
        else:
            features = data
            targets = torch.zeros (len (data))  # Placeholder targets

        return TensorDataset (features, targets)


def _generate_test_data(pl_module):
    """Generate structured test data for visualization."""
    num_samples = 100
    input_dim = pl_module.config ['model'] ['input_dim']
    output_dim = pl_module.config ['model'] ['output_dim']

    # Create structured features
    features = []
    labels = []

    for i in range (output_dim):
        # Generate samples for each class with some pattern
        class_samples = num_samples // output_dim

        # Create features with class-specific patterns
        class_features = torch.randn (class_samples, input_dim)
        # Add class-specific bias to make features more separable
        class_features [:, 0] += i * 2

        features.append (class_features)
        labels.extend ([i] * class_samples)

    features = torch.cat (features)
    labels = torch.tensor (labels)

    return features, labels


def _get_activation(activation_name):
    """Get the activation function based on the name."""
    if activation_name == 'relu':
        return nn.ReLU ()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid ()
    elif activation_name == 'tanh':
        return nn.Tanh ()
    else:
        raise ValueError (f"Unsupported activation function: {activation_name}")


def _prepare_data(data):
    """Convert input data to PyTorch Dataset."""
    if isinstance (data, np.ndarray):
        data = torch.from_numpy (data).float ()
    elif not isinstance (data, torch.Tensor):
        raise ValueError (f"Unsupported data type: {type (data)}")

    if len (data.shape) != 2:
        raise ValueError ("Data must be 2-dimensional")

    features = data [:, :-1]
    targets = data [:, -1]

    return TensorDataset (features, targets)


def _prepare_data(data):
    """Convert input data to PyTorch Dataset."""
    if isinstance (data, np.ndarray):
        data = torch.from_numpy (data).float ()
    elif not isinstance (data, torch.Tensor):
        raise ValueError (f"Unsupported data type: {type (data)}")

    if len (data.shape) != 2:
        raise ValueError ("Data must be 2-dimensional")

    features = data [:, :-1]
    targets = data [:, -1]

    return TensorDataset (features, targets)


def _prepare_data(data):
    if isinstance (data, np.ndarray):
        features = torch.FloatTensor (data [:, :-1])
        targets = torch.FloatTensor (data [:, -1])
        return TensorDataset (features, targets)
    raise ValueError ("Data must be numpy array")


class AutoML:
    def __init__(self, config: Dict [str, Any]):
        self.config = config
        self.model = self._create_model ()

    def _create_model(self):
        model_type = self.config ['model'] ['type']
        if model_type == 'logistic_regression':
            return LogisticRegressionModel (self.config)
        elif model_type == 'linear_regression':
            return LinearRegressionModel (self.config)
        else:
            raise ValueError (f"Unsupported model type: {model_type}")

    def fit(self, train_data, val_data=None):
        # Prepare datasets
        train_dataset = _prepare_data (train_data)
        val_dataset = _prepare_data (val_data) if val_data is not None else None

        # Create dataloaders
        train_loader = DataLoader (
            train_dataset,
            batch_size=self.config.get ('batch_size', 32),
            shuffle=True
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader (
                val_dataset,
                batch_size=self.config.get ('batch_size', 32)
            )

        # Train model
        trainer = pl.Trainer (
            max_epochs=self.config.get ('epochs', 10),
            accelerator="auto",
            devices="auto",
            enable_progress_bar=True
        )

        trainer.fit (self.model, train_loader, val_loader)
        return self.model


def _compute_loss(self, batch):
    """Compute classification loss with proper handling."""
    features, targets = batch
    outputs = self.model (features)

    if self.config ['model'] ['task'] == 'classification':
        # Convert targets to long for classification
        targets = targets.long ()
        loss = nn.CrossEntropyLoss () (outputs, targets)

        # Compute and log accuracy
        with torch.no_grad ():
            predictions = outputs.argmax (dim=1)
            accuracy = (predictions == targets).float ().mean ()
            self.log ('accuracy', accuracy, prog_bar=True)
    else:
        loss = nn.MSELoss () (outputs, targets)

    return loss


def configure_optimizers(self):
    """Configure optimizer and scheduler with proper step calculation."""
    opt_config = self.config.get ('optimization', {})

    # Setup optimizer
    optimizer = torch.optim.Adam (
        self.parameters (),
        lr=self.config ['training'].get ('learning_rate', 0.001),
        weight_decay=opt_config.get ('weight_decay', 0.01)
    )

    # Setup scheduler
    scheduler_config = {
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR (
            optimizer,
            T_max=self.config ['training'].get ('epochs', 30),
            eta_min=opt_config.get ('min_lr', 1e-6)
        ),
        "interval": "epoch",  # Changed to epoch-based scheduling
        "frequency": 1,
        "monitor": "val_loss"
    }

    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler_config
    }


def training_step(self, batch, batch_idx):
    """Training step with proper loss computation."""
    with autocast (enabled=self.use_amp):
        loss = self._compute_loss (batch)

    # Log metrics
    self.log ('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log ('lr', self.optimizers ().param_groups [0] ['lr'], on_step=False, on_epoch=True, prog_bar=True)

    return loss


def validation_step(self, batch, batch_idx):
    """Validation step with proper loss computation."""
    with autocast (enabled=self.use_amp):
        loss = self._compute_loss (batch)

    self.log ('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    return loss


def training_step(self, batch, batch_idx):
    """Optimized training step."""
    with autocast (enabled=self.use_amp):
        loss = self._compute_loss (batch)

    # Log metrics
    self.log ('train_loss', loss, prog_bar=True)
    self.log ('lr', self.optimizers ().param_groups [0] ['lr'], prog_bar=True)

    return loss


def validation_step(self, batch, batch_idx):
    """Optimized validation step."""
    with autocast (enabled=self.use_amp):
        loss = self._compute_loss (batch)

    self.log ('val_loss', loss, prog_bar=True)
    return loss


def _get_callbacks(self):
    """Get enhanced callbacks including visualizations."""
    callbacks = [self.VisualizationCallback (), self.MetricsCallback ()]

    # Early stopping with improved messaging
    if self.config ['training'].get ('early_stopping', False):
        from pytorch_lightning.callbacks import EarlyStopping
        callbacks.append (EarlyStopping (
            monitor='val_loss',
            patience=5,
            mode='min',
            verbose=True
        ))

    # Model checkpoint with improved config
    if self.config ['training'].get ('model_checkpoint', True):
        from pytorch_lightning.callbacks import ModelCheckpoint
        callbacks.append (ModelCheckpoint (
            monitor='val_loss',
            filename='model-{epoch:02d}-{val_loss:.3f}',
            save_top_k=3,
            mode='min',
            verbose=True
        ))

    return callbacks


def _get_logger(self):
    """Get logger based on configuration."""
    if self.config.get ('logging', {}).get ('tensorboard', True):
        from pytorch_lightning.loggers import TensorBoardLogger
        return TensorBoardLogger ("lightning_logs/")
    return True  # Default logger


class MetricsCallback (Callback):
    """Custom callback for detailed metrics logging."""

    def __init__(self):
        super ().__init__ ()
        self.epoch_metrics = {}

    def on_train_epoch_start(self, trainer, pl_module):
        """Initialize metrics for new epoch."""
        print (f"\nEpoch {trainer.current_epoch + 1}/{trainer.max_epochs}")
        print ("-" * 50)

    def on_train_epoch_end(self, trainer, pl_module):
        """Log detailed metrics at end of epoch."""
        metrics = {
            'Training Loss': f"{trainer.callback_metrics.get ('train_loss', 0):.4f}",
            'Validation Loss': f"{trainer.callback_metrics.get ('val_loss', 0):.4f}",
            'Learning Rate': f"{trainer.optimizers [0].param_groups [0] ['lr']:.6f}"
        }

        print ("\nEpoch Summary:")
        for name, value in metrics.items ():
            print (f"{name}: {value}")
        print ("-" * 50)
