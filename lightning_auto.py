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

        basic_stats = _compute_basic_statistics (features, feature_names)
        correlations = self._analyze_correlations (features, target, feature_names)
        distributions = _analyze_distributions (features, feature_names)
        target_analysis = _analyze_target (target)
        missing_values = _analyze_missing_values (features, feature_names)
        outliers = _detect_outliers (features, feature_names)

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
        if len (data.shape) == 2:
            features = data [:, :-1]
            targets = data [:, -1]
        else:
            features = data
            targets = torch.zeros (len (data))

        return TensorDataset (features, targets)


def _generate_test_data(pl_module):
    """Generate structured test data for visualization."""
    num_samples = 100
    input_dim = pl_module.config ['model'] ['input_dim']
    output_dim = pl_module.config ['model'] ['output_dim']

    features = []
    labels = []

    for i in range (output_dim):
        class_samples = num_samples // output_dim

        class_features = torch.randn (class_samples, input_dim)
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
