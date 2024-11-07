import torch
import torch.nn as nn
import pytorch_lightning as pl
from visualization import LogisticRegressionVisualizer, LinearRegressionVisualizer
from typing import List, Dict, Any


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

    def configure_callbacks(self) -> List:
        callbacks = [LogisticRegressionVisualizer ()]
        return callbacks


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
