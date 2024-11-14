# visualization.py

import os
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import confusion_matrix, roc_curve, auc


class LogisticRegressionVisualizer (Callback):
    """Enhanced visualization callback for logistic regression."""

    def __init__(self, log_dir: str = "training_plots/logistic_regression"):
        super ().__init__ ()
        self.log_dir = log_dir
        os.makedirs (log_dir, exist_ok=True)
        self.predictions = []
        self.actuals = []
        self.probabilities = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.training_losses = []
        self.validation_losses = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        """Collect predictions and actual values.
        """
        x, y = batch
        with torch.no_grad ():
            probs = pl_module (x)
            preds = (probs > 0.5).float ()

            self.predictions.extend (preds.cpu ().numpy ().flatten ())
            self.actuals.extend (y.cpu ().numpy ().flatten ())
            self.probabilities.extend (probs.cpu ().numpy ().flatten ())

    def on_train_epoch_end(self, trainer, pl_module):
        """Plot visualizations at the end of each epoch."""
        metrics = trainer.callback_metrics

        # Collect metrics
        self.accuracies.append (metrics.get ('accuracy', 0).item ())
        self.precisions.append (metrics.get ('precision', 0).item ())
        self.recalls.append (metrics.get ('recall', 0).item ())
        self.training_losses.append (metrics.get ('train_loss', 0).item ())
        self.validation_losses.append (metrics.get ('val_loss', 0).item ())

        # Create plots
        self._plot_metrics_over_time ()
        self._plot_roc_curve ()
        self._plot_confusion_matrix ()
        self._plot_probability_distribution ()

        # Clear batch data
        self.predictions = []
        self.actuals = []
        self.probabilities = []

    def _plot_metrics_over_time(self):
        """Plot accuracy, precision, recall, and F1 score over time."""
        plt.figure (figsize=(12, 6))
        epochs = range (1, len (self.accuracies) + 1)

        plt.plot (epochs, self.accuracies, 'b-', label='Accuracy')
        plt.plot (epochs, self.precisions, 'g-', label='Precision')
        plt.plot (epochs, self.recalls, 'r-', label='Recall')

        plt.title ('Classification Metrics Over Time')
        plt.xlabel ('Epoch')
        plt.ylabel ('Score')
        plt.legend ()
        plt.grid (True)
        plt.savefig (os.path.join (self.log_dir, 'metrics_over_time.png'))
        plt.close ()

    def _plot_roc_curve(self):
        """Plot ROC curve and calculate AUC."""
        if len (self.actuals) > 0:
            fpr, tpr, _ = roc_curve (self.actuals, self.probabilities)
            roc_auc = auc (fpr, tpr)

            plt.figure (figsize=(8, 8))
            plt.plot (fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot ([0, 1], [0, 1], 'k--')
            plt.xlim ([0.0, 1.0])
            plt.ylim ([0.0, 1.05])
            plt.xlabel ('False Positive Rate')
            plt.ylabel ('True Positive Rate')
            plt.title ('Receiver Operating Characteristic (ROC)')
            plt.legend (loc="lower right")
            plt.grid (True)
            plt.savefig (os.path.join (self.log_dir, 'roc_curve.png'))
            plt.close ()

    def _plot_confusion_matrix(self):
        """Plot confusion matrix heatmap."""
        if len (self.actuals) > 0:
            cm = confusion_matrix (self.actuals, self.predictions)
            plt.figure (figsize=(8, 6))
            plt.imshow (cm, interpolation='nearest', cmap='Blues')
            plt.title ('Confusion Matrix')
            plt.colorbar ()

            classes = ['Negative', 'Positive']
            tick_marks = np.arange (len (classes))
            plt.xticks (tick_marks, classes)
            plt.yticks (tick_marks, classes)

            # Add text annotations to the matrix
            for i in range (cm.shape [0]):
                for j in range (cm.shape [1]):
                    plt.text (j, i, str (cm [i, j]),
                              horizontalalignment="center",
                              color="white" if cm [i, j] > cm.max () / 2 else "black")

            plt.xlabel ('Predicted label')
            plt.ylabel ('True label')
            plt.tight_layout ()
            plt.savefig (os.path.join (self.log_dir, 'confusion_matrix.png'))
            plt.close ()

    def _plot_probability_distribution(self):
        """Plot distribution of predicted probabilities."""
        if len (self.probabilities) > 0:
            plt.figure (figsize=(10, 6))
            plt.hist (self.probabilities, bins=50, edgecolor='black')
            plt.title ('Distribution of Predicted Probabilities')
            plt.xlabel ('Predicted Probability')
            plt.ylabel ('Count')
            plt.grid (True)
            plt.savefig (os.path.join (self.log_dir, 'probability_distribution.png'))
            plt.close ()


class LinearRegressionVisualizer (Callback):
    """Enhanced visualization callback for linear regression."""

    def __init__(self, log_dir: str = "training_plots/linear_regression"):
        super ().__init__ ()
        self.log_dir = log_dir
        os.makedirs (log_dir, exist_ok=True)
        self.predictions = []
        self.actuals = []
        self.mse_scores = []
        self.r2_scores = []
        self.training_losses = []
        self.validation_losses = []
        self.residuals = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        """Collect predictions and actual values.
        """
        x, y = batch
        with torch.no_grad ():
            preds = pl_module (x)

            self.predictions.extend (preds.cpu ().numpy ().flatten ())
            self.actuals.extend (y.cpu ().numpy ().flatten ())
            self.residuals.extend ((preds - y).cpu ().numpy ().flatten ())

    def on_train_epoch_end(self, trainer, pl_module):
        """Plot visualizations at the end of each epoch."""
        metrics = trainer.callback_metrics

        # Collect metrics
        self.mse_scores.append (metrics.get ('mse', 0).item ())
        self.r2_scores.append (metrics.get ('r2_score', 0).item ())
        self.training_losses.append (metrics.get ('train_loss', 0).item ())
        self.validation_losses.append (metrics.get ('val_loss', 0).item ())

        # Create plots
        self._plot_metrics_over_time ()
        self._plot_residuals ()
        self._plot_predictions_vs_actual ()
        self._plot_residual_distribution ()

        # Clear batch data
        self.predictions = []
        self.actuals = []
        self.residuals = []

    def _plot_metrics_over_time(self):
        """Plot MSE and R² score over time."""
        plt.figure (figsize=(12, 6))
        epochs = range (1, len (self.mse_scores) + 1)

        fig, (ax1, ax2) = plt.subplots (1, 2, figsize=(15, 5))

        # Plot MSE
        ax1.plot (epochs, self.mse_scores, 'r-')
        ax1.set_title ('Mean Squared Error Over Time')
        ax1.set_xlabel ('Epoch')
        ax1.set_ylabel ('MSE')
        ax1.grid (True)

        # Plot R²
        ax2.plot (epochs, self.r2_scores, 'b-')
        ax2.set_title ('R² Score Over Time')
        ax2.set_xlabel ('Epoch')
        ax2.set_ylabel ('R²')
        ax2.grid (True)

        plt.tight_layout ()
        plt.savefig (os.path.join (self.log_dir, 'metrics_over_time.png'))
        plt.close ()

    def _plot_residuals(self):
        """Plot residuals vs predicted values."""
        if len (self.predictions) > 0:
            plt.figure (figsize=(10, 6))
            plt.scatter (self.predictions, self.residuals, alpha=0.5)
            plt.axhline (y=0, color='r', linestyle='--')
            plt.title ('Residual Plot')
            plt.xlabel ('Predicted Values')
            plt.ylabel ('Residuals')
            plt.grid (True)
            plt.savefig (os.path.join (self.log_dir, 'residuals.png'))
            plt.close ()

    def _plot_predictions_vs_actual(self):
        """Plot predicted vs actual values."""
        if len (self.predictions) > 0:
            plt.figure (figsize=(10, 6))
            plt.scatter (self.actuals, self.predictions, alpha=0.5)

            # Add perfect prediction line
            min_val = min (min (self.actuals), min (self.predictions))
            max_val = max (max (self.actuals), max (self.predictions))
            plt.plot ([min_val, max_val], [min_val, max_val], 'r--')

            plt.title ('Predictions vs Actual Values')
            plt.xlabel ('Actual Values')
            plt.ylabel ('Predicted Values')
            plt.grid (True)
            plt.savefig (os.path.join (self.log_dir, 'predictions_vs_actual.png'))
            plt.close ()

    def _plot_residual_distribution(self):
        """Plot distribution of residuals."""
        if len (self.residuals) > 0:
            plt.figure (figsize=(10, 6))
            plt.hist (self.residuals, bins=50, edgecolor='black')
            plt.title ('Distribution of Residuals')
            plt.xlabel ('Residual Value')
            plt.ylabel ('Count')
            plt.grid (True)
            plt.savefig (os.path.join (self.log_dir, 'residual_distribution.png'))
            plt.close ()
