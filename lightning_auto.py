import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast
import torch.nn as nn
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import Callback
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
import logging
from pathlib import Path


@dataclass
class EDAResults:
    """Container for EDA results to ensure type safety and clear data structure."""
    basic_stats: pd.DataFrame
    correlations: Dict [str, np.ndarray]
    distributions: Dict [str, Dict [str, float]]
    target_analysis: Dict [str, Any]
    missing_values: Dict [str, List [int]]
    outliers: Dict [str, List [int]]
    visualization_paths: Dict [str, str]


def _analyze_missing_values(features: np.ndarray, feature_names: List [str]) -> Dict [str, List [int]]:
    """Analyzes missing values in the dataset."""
    missing_dict = {}
    for i, name in enumerate (feature_names):
        missing_indices = np.where (np.isnan (features [:, i])) [0].tolist ()
        missing_dict [name] = missing_indices
    return missing_dict


def _detect_outliers(features: np.ndarray, feature_names: List [str]) -> Dict [str, List [int]]:
    """Detects outliers using the IQR method."""
    outliers_dict = {}

    for i, name in enumerate (feature_names):
        feature_data = features [:, i]
        Q1 = np.percentile (feature_data, 25)
        Q3 = np.percentile (feature_data, 75)
        IQR = Q3 - Q1

        outlier_mask = (feature_data < (Q1 - 1.5 * IQR)) | (feature_data > (Q3 + 1.5 * IQR))
        outliers_dict [name] = np.where (outlier_mask) [0].tolist ()

    return outliers_dict


def _analyze_target(target: np.ndarray) -> Dict [str, Any]:
    """Analyzes the target variable characteristics."""
    unique_values = np.unique (target)
    is_classification = len (unique_values) < 10

    if is_classification:
        class_counts = pd.Series (target).value_counts ().to_dict ()
        class_proportions = pd.Series (target).value_counts (normalize=True).to_dict ()
    else:
        class_counts = None
        class_proportions = None

    return {
        'is_classification': is_classification,
        'unique_values': len (unique_values),
        'class_counts': class_counts,
        'class_proportions': class_proportions,
        'mean': float (np.mean (target)),
        'std': float (np.std (target))
    }


def _analyze_distributions(features: np.ndarray, feature_names: List [str]) -> Dict [str, Dict [str, float]]:
    """Analyzes the distribution characteristics of each feature."""
    distribution_tests = {}

    for i, name in enumerate (feature_names):
        _, p_value = stats.normaltest (features [:, i])
        distribution_tests [name] = {
            'normality_p_value': p_value,
            'is_normal': p_value > 0.05,
            'skewness': float (stats.skew (features [:, i])),
            'kurtosis': float (stats.kurtosis (features [:, i]))
        }

    return distribution_tests


def _compute_basic_statistics(features: np.ndarray, feature_names: List [str]) -> pd.DataFrame:
    """Computes basic statistical measures for features."""
    df = pd.DataFrame (features, columns=feature_names)
    return df.describe ().round (4)


class DataAnalysis:
    """Handles data analysis within the AutoML pipeline."""

    def __init__(self, config: Dict [str, Any], output_dir: Optional [str] = None):
        self.config = config
        self.output_dir = Path (output_dir) if output_dir else Path ("eda_outputs")
        self.output_dir.mkdir (parents=True, exist_ok=True)
        self.logger = logging.getLogger (__name__)

    def analyze(self, data: np.ndarray, feature_names: Optional [List [str]] = None) -> EDAResults:
        """
        Performs comprehensive data analysis on the input dataset.

        Args:
            data: Input data array where last column is the target
            feature_names: Optional list of feature names
        """
        if not isinstance (data, np.ndarray):
            data = np.array (data)

        features = data [:, :-1]
        target = data [:, -1]

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range (features.shape [1])]

        # Compute all analyses
        basic_stats = _compute_basic_statistics (features, feature_names)
        correlations = self._analyze_correlations (features, target, feature_names)
        distributions = _analyze_distributions (features, feature_names)
        target_analysis = _analyze_target (target)
        missing_values = _analyze_missing_values (features, feature_names)
        outliers = _detect_outliers (features, feature_names)

        # Generate all visualizations
        visualization_paths = self._generate_visualizations (features, target, feature_names)

        return EDAResults (
            basic_stats=basic_stats,
            correlations=correlations,
            distributions=distributions,
            target_analysis=target_analysis,
            missing_values=missing_values,
            outliers=outliers,
            visualization_paths=visualization_paths
        )

    def _analyze_correlations(self, features: np.ndarray, target: np.ndarray,
                              feature_names: List [str]) -> Dict [str, np.ndarray]:
        """Analyzes feature correlations and feature-target relationships."""
        df = pd.DataFrame (np.column_stack ([features, target]),
                           columns=feature_names + ['target'])
        correlation_matrix = df.corr ().round (4)

        # Save correlation heatmap
        plt.figure (figsize=(10, 8))
        sns.heatmap (correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title ('Feature Correlation Heatmap')
        plt.tight_layout ()
        plt.savefig (self.output_dir / 'correlation_heatmap.png')
        plt.close ()

        return {
            'correlation_matrix': correlation_matrix.values,
            'feature_target_correlations': correlation_matrix ['target'].drop ('target').values
        }

    def _generate_visualizations(self, features: np.ndarray, target: np.ndarray,
                                 feature_names: List [str]) -> Dict [str, str]:
        """Generates and saves all visualization plots."""
        visualization_paths = {}

        # Feature distributions
        fig, axes = plt.subplots (len (feature_names), 1, figsize=(10, 5 * len (feature_names)))
        if len (feature_names) == 1:
            axes = [axes]

        for ax, name, feature in zip (axes, feature_names, features.T):
            sns.histplot (feature, kde=True, ax=ax)
            ax.set_title (f'{name} Distribution')

        plt.tight_layout ()
        dist_path = self.output_dir / 'feature_distributions.png'
        plt.savefig (dist_path)
        plt.close ()
        visualization_paths ['distributions'] = str (dist_path)

        # Feature-target relationships
        fig, axes = plt.subplots (len (feature_names), 1, figsize=(10, 5 * len (feature_names)))
        if len (feature_names) == 1:
            axes = [axes]

        for ax, name, feature in zip (axes, feature_names, features.T):
            ax.scatter (feature, target, alpha=0.5)
            ax.set_xlabel (name)
            ax.set_ylabel ('Target')
            ax.set_title (f'{name} vs Target')

        plt.tight_layout ()
        scatter_path = self.output_dir / 'feature_target_relationships.png'
        plt.savefig (scatter_path)
        plt.close ()
        visualization_paths ['scatter_plots'] = str (scatter_path)

        return visualization_paths


def _convert_to_dataset(data):
    """Convert raw data to PyTorch Dataset."""
    if isinstance (data, np.ndarray):
        data = torch.from_numpy (data)

    if isinstance (data, torch.Tensor):
        # Assuming the last column is target for structured data
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
