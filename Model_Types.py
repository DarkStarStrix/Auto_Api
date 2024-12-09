import torch
import torch.nn as nn
import pytorch_lightning as pl
from visualization import NaiveBayesVisualizer, DecisionTreeVisualizer
from typing import List, Dict, Any
import numpy as np


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
