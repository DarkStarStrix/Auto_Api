import torch
import torch.nn as nn
import pytorch_lightning as pl
from visualization import NaiveBayesVisualizer, DecisionTreeVisualizer, KMeansVisualizer, LightGBMVisualizer
from typing import List, Dict, Any
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import lightgbm as lgb


class BaseModel (pl.LightningModule):
    def __init__(self, config: Dict [str, Any]):
        super ().__init__ ()
        self.config = config
        self.save_hyperparameters (config)
        self.model = self._create_model ()

    def forward(self, x):
        return self.model (x)

    def configure_optimizers(self):
        return torch.optim.Adam (
            self.parameters (),
            lr=self.config ['training'].get ('learning_rate', 0.001)
        )

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss (batch)
        self.log ('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss (batch)
        self.log ('val_loss', loss, prog_bar=True)
        return loss


class LinearRegressionModel (BaseModel):
    def _create_model(self):
        input_dim = self.config ['model'] ['input_dim']
        return nn.Linear (input_dim, 1)

    def _compute_loss(self, batch):
        x, y = batch
        # Ensure proper dimensions
        x = x.float ().view (x.size (0), -1)  # Flatten any input dimensions
        y = y.float ().view (-1, 1)  # Reshape target to [batch_size, 1]

        # Forward pass
        y_hat = self (x)

        # Compute MSE loss
        loss = nn.MSELoss () (y_hat, y)

        # Compute metrics
        with torch.no_grad ():
            # MSE
            self.log ('mse', loss, prog_bar=True)

            # R² score
            y_mean = torch.mean (y)
            ss_tot = torch.sum ((y - y_mean) ** 2)
            ss_res = torch.sum ((y - y_hat) ** 2)
            r2 = 1 - ss_res / ss_tot
            self.log ('r2_score', r2, prog_bar=True)

        return loss

    def configure_callbacks(self) -> List:
        callbacks = [LinearRegressionVisualizer ()]
        return callbacks


class LogisticRegressionModel (BaseModel):
    def _create_model(self):
        input_dim = self.config ['model'] ['input_dim']
        return nn.Sequential (
            nn.Linear (input_dim, 1),
            nn.Sigmoid ()
        )

    def _compute_loss(self, batch):
        x, y = batch
        # Ensure proper dimensions
        x = x.float ().view (x.size (0), -1)  # Flatten any input dimensions
        y = y.float ().view (-1, 1)  # Reshape target to [batch_size, 1]

        # Forward pass
        y_hat = self (x)

        # Compute loss
        loss = nn.BCELoss () (y_hat, y)

        # Compute metrics
        with torch.no_grad ():
            predictions = (y_hat > 0.5).float ()
            accuracy = (predictions == y).float ().mean ()

            # Compute precision and recall
            true_positives = ((predictions == 1) & (y == 1)).float ().sum ()
            predicted_positives = (predictions == 1).float ().sum ()
            actual_positives = (y == 1).float ().sum ()

            precision = true_positives / predicted_positives if predicted_positives > 0 else torch.tensor (0.0)
            recall = true_positives / actual_positives if actual_positives > 0 else torch.tensor (0.0)

            # Log metrics
            self.log ('accuracy', accuracy, prog_bar=True)
            self.log ('precision', precision, prog_bar=True)
            self.log ('recall', recall, prog_bar=True)

        return loss


class NaiveBayesModel (BaseModel):
    def __init__(self, config: Dict [str, Any]):
        super ().__init__ (config)
        self.input_dim = config ['model'] ['input_dim']
        self.output_dim = config ['model'] ['output_dim']

        # Parameters for Gaussian NB
        self.class_priors = nn.Parameter (torch.zeros (self.output_dim))
        self.feature_means = nn.Parameter (torch.zeros (self.output_dim, self.input_dim))
        self.feature_vars = nn.Parameter (torch.ones (self.output_dim, self.input_dim))
        self.var_smoothing = config ['model'].get ('var_smoothing', 1e-9)

    def forward(self, x):
        x = x.view (-1, self.input_dim)
        log_probs = torch.zeros (x.size (0), self.output_dim, device=x.device)

        for i in range (self.output_dim):
            diff = x - self.feature_means [i]
            var = self.feature_vars [i] + self.var_smoothing
            log_prob = -0.5 * torch.sum (
                torch.log (2 * np.pi * var) + (diff ** 2) / var,
                dim=1
            )
            log_probs [:, i] = log_prob + self.class_priors [i]

        return torch.softmax (log_probs, dim=1)

    def _compute_loss(self, batch):
        x, y = batch
        y_hat = self (x)
        loss = nn.CrossEntropyLoss () (y_hat, y.long ())

        with torch.no_grad ():
            predictions = torch.argmax (y_hat, dim=1)
            for i in range (self.output_dim):
                true_pos = ((predictions == i) & (y == i)).float ().sum ()
                pred_pos = (predictions == i).float ().sum ()
                actual_pos = (y == i).float ().sum ()

                precision = true_pos / pred_pos if pred_pos > 0 else torch.tensor (0.0)
                recall = true_pos / actual_pos if actual_pos > 0 else torch.tensor (0.0)

                self.log (f'precision_class_{i}', precision, prog_bar=True)
                self.log (f'recall_class_{i}', recall, prog_bar=True)

        return loss

    def configure_callbacks(self) -> List:
        return [NaiveBayesVisualizer ()]


def _create_model():
    return nn.Identity ()


class DecisionTreeModel (pl.LightningModule):
    def __init__(self, config: Dict [str, Any]):
        super ().__init__ ()
        self.input_dim = config ['model'] ['input_dim']
        self.output_dim = config ['model'] ['output_dim']
        self.max_depth = config ['model'].get ('max_depth', 5)
        self.learning_rate = config ['training'].get ('learning_rate', 0.001)

        self.feature_thresholds = nn.Parameter (torch.randn (2 ** self.max_depth - 1, self.input_dim))
        self.leaf_predictions = nn.Parameter (torch.randn (2 ** self.max_depth, self.output_dim))

        self.feature_importance = None
        self.n_features = self.input_dim

    def _traverse_tree(self, x, node_idx=0, depth=0):
        if depth >= self.max_depth or node_idx >= 2 ** self.max_depth - 1:
            return self.leaf_predictions [node_idx - (2 ** self.max_depth - 1)]

        decision = torch.sigmoid (torch.matmul (x, self.feature_thresholds [node_idx]))
        left_idx = 2 * node_idx + 1
        right_idx = 2 * node_idx + 2

        left_result = self._traverse_tree (x, left_idx, depth + 1)
        right_result = self._traverse_tree (x, right_idx, depth + 1)

        return decision.unsqueeze (-1) * left_result + (1 - decision).unsqueeze (-1) * right_result

    def forward(self, x):
        x = x.view (-1, self.input_dim)
        predictions = self._traverse_tree (x)
        return torch.softmax (predictions, dim=1)

    def training_step(self, batch, batch_idx):
        return self._compute_loss (batch)

    def validation_step(self, batch, batch_idx):
        return self._compute_loss (batch)

    def _compute_loss(self, batch):
        x, y = batch
        y_hat = self (x)
        loss = nn.CrossEntropyLoss () (y_hat, y.long ())

        with torch.no_grad ():
            predictions = torch.argmax (y_hat, dim=1)
            accuracy = (predictions == y).float ().mean ()
            self.log ('accuracy', accuracy, prog_bar=True)

            importance = torch.abs (self.feature_thresholds).mean (dim=0)
            self.feature_importance = importance / importance.sum ()

            for i, imp in enumerate (self.feature_importance):
                self.log (f'feature_importance_{i}', imp)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam (self.parameters (), lr=self.learning_rate)

    def configure_callbacks(self):
        return [DecisionTreeVisualizer ()]


class KMeansModel (pl.LightningModule):
    def __init__(self, config: Dict [str, Any]):
        super ().__init__ ()
        self.input_dim = config ['model'] ['input_dim']
        self.n_clusters = config ['model'] ['n_clusters']
        self.learning_rate = config ['training'].get ('learning_rate', 0.001)
        self.batch_size = config ['training'].get ('batch_size', 32)

        # Initialize centroids as learnable parameters
        self.centroids = nn.Parameter (torch.randn (self.n_clusters, self.input_dim))

        # Initialize trackers
        self.cluster_sizes = None
        self.inertia = None
        self.n_features = self.input_dim

        # Data storage
        self.train_data = None
        self.val_data = None

    def _compute_distances(self, x):
        # Compute Euclidean distances between points and centroids
        x_expanded = x.unsqueeze (1)  # Shape: (batch_size, 1, input_dim)
        centroids_expanded = self.centroids.unsqueeze (0)  # Shape: (1, n_clusters, input_dim)
        distances = torch.norm (x_expanded - centroids_expanded, dim=2)
        return distances

    def forward(self, x):
        x = x.view (-1, self.input_dim)
        distances = self._compute_distances (x)
        assignments = torch.argmin (distances, dim=1)
        return distances, assignments

    def training_step(self, batch, batch_idx):
        return self._compute_loss (batch)

    def validation_step(self, batch, batch_idx):
        return self._compute_loss (batch)

    def _compute_loss(self, batch):
        x, _ = batch
        distances, assignments = self (x)

        # Compute inertia (sum of squared distances to nearest centroid)
        min_distances = torch.min (distances, dim=1) [0]
        loss = torch.mean (min_distances ** 2)

        with torch.no_grad ():
            # Update inertia
            self.inertia = loss.item ()
            self.log ('inertia', self.inertia, prog_bar=True)

            # Compute and log cluster sizes
            unique_clusters, counts = torch.unique (assignments, return_counts=True)
            sizes = torch.zeros (self.n_clusters, device=self.device)
            sizes [unique_clusters] = counts.float ()
            self.cluster_sizes = sizes / len (assignments)

            for i, size in enumerate (self.cluster_sizes):
                self.log (f'cluster_size_{i}', size)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam (self.parameters (), lr=self.learning_rate)

    def configure_callbacks(self):
        return [KMeansVisualizer ()]

    def set_train_data(self, X: torch.Tensor):
        """Set training data"""
        self.train_data = X

    def set_val_data(self, X: torch.Tensor):
        """Set validation data"""
        self.val_data = X

    def train_dataloader(self):
        """Return training dataloader"""
        if self.train_data is None:
            raise ValueError ("Training data not set. Call set_train_data first.")
        dataset = TensorDataset (self.train_data, self.train_data)  # Use same data for X and y
        return DataLoader (dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Return validation dataloader"""
        if self.val_data is None:
            return None
        dataset = TensorDataset (self.val_data, self.val_data)  # Use same data for X and y
        return DataLoader (dataset, batch_size=self.batch_size, shuffle=False)


class LightGBMModel (pl.LightningModule):
    def __init__(self, config: Dict [str, Any]):
        super ().__init__ ()
        self.automatic_optimization = False  # Turn off automatic optimization

        self.input_dim = config ['model'] ['input_dim']
        self.output_dim = config ['model'] ['output_dim']
        self.learning_rate = config ['training'].get ('learning_rate', 0.01)
        self.num_leaves = config ['model'].get ('num_leaves', 31)

        # Initialize model
        self.model = None
        self.feature_importance = None

        # Model parameters
        self.params = {
            'objective': 'multiclass' if self.output_dim > 2 else 'binary',
            'num_class': self.output_dim if self.output_dim > 2 else None,
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'verbose': -1
        }

    def forward(self, x):
        if self.model is None:
            return torch.zeros (x.size (0), self.output_dim, device=self.device)

        # Get predictions
        x_np = x.cpu ().numpy ()
        preds = self.model.predict (x_np)

        # Convert to tensor
        if self.output_dim > 2:
            return torch.from_numpy (preds).float ().to (self.device)
        else:
            preds = torch.from_numpy (preds).float ().unsqueeze (1).to (self.device)
            return torch.cat ([1 - preds, preds], dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_np = x.cpu ().numpy ()
        y_np = y.cpu ().numpy ()

        # Train LightGBM model
        if self.model is None:
            train_data = lgb.Dataset (x_np, y_np)
            self.model = lgb.train (self.params, train_data, num_boost_round=1)
        else:
            train_data = lgb.Dataset (x_np, y_np)
            self.model = lgb.train (
                self.params,
                train_data,
                num_boost_round=1,
                init_model=self.model
            )

        # Get predictions for logging
        y_hat = self (x)
        predictions = torch.argmax (y_hat, dim=1)
        accuracy = (predictions == y).float ().mean ()

        # Log metrics
        self.log ('train_accuracy', accuracy, prog_bar=True)

        if self.model is not None:
            importance = self.model.feature_importance ()
            importance = importance / importance.sum ()
            for i, imp in enumerate (importance):
                self.log (f'feature_importance_{i}', imp)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self (x)

        predictions = torch.argmax (y_hat, dim=1)
        accuracy = (predictions == y).float ().mean ()
        self.log ('val_accuracy', accuracy, prog_bar=True)

    def configure_optimizers(self):
        # No optimizer needed
        return None

    def configure_callbacks(self):
        return [LightGBMVisualizer ()]
