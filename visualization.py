# visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class LogisticRegressionVisualizer (Callback):
    """Visualization callback for logistic regression."""

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
        self.training_losses = []
        self.validation_losses = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        """Collect predictions and actual values."""
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
        self._plot_probability_distribution ()
        self._plot_custom_confusion_matrix ()

        # Clear batch data
        self.predictions = []
        self.actuals = []
        self.probabilities = []

    def _plot_metrics_over_time(self):
        """Plot accuracy, precision, and recall over time."""
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

    def _plot_custom_confusion_matrix(self):
        """Plot simplified confusion matrix."""
        if len (self.actuals) > 0:
            # Calculate confusion matrix manually
            predictions = np.array (self.predictions)
            actuals = np.array (self.actuals)

            tp = np.sum ((predictions == 1) & (actuals == 1))
            tn = np.sum ((predictions == 0) & (actuals == 0))
            fp = np.sum ((predictions == 1) & (actuals == 0))
            fn = np.sum ((predictions == 0) & (actuals == 1))

            cm = np.array ([[tn, fp], [fn, tp]])

            plt.figure (figsize=(8, 6))
            plt.imshow (cm, interpolation='nearest', cmap='Blues')
            plt.title ('Confusion Matrix')
            plt.colorbar ()

            classes = ['Negative', 'Positive']
            tick_marks = np.arange (len (classes))
            plt.xticks (tick_marks, classes)
            plt.yticks (tick_marks, classes)

            # Add text annotations
            for i in range (2):
                for j in range (2):
                    plt.text (j, i, str (cm [i, j]),
                              horizontalalignment="center",
                              color="white" if cm [i, j] > cm.max () / 2 else "black")

            plt.xlabel ('Predicted label')
            plt.ylabel ('True label')
            plt.tight_layout ()
            plt.savefig (os.path.join (self.log_dir, 'confusion_matrix.png'))
            plt.close ()


class LinearRegressionVisualizer (Callback):
    """Visualization callback for linear regression."""

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
        """Collect predictions and actual values."""
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
        plt.figure (figsize=(15, 5))
        epochs = range (1, len (self.mse_scores) + 1)

        plt.subplot (1, 2, 1)
        plt.plot (epochs, self.mse_scores, 'r-')
        plt.title ('Mean Squared Error Over Time')
        plt.xlabel ('Epoch')
        plt.ylabel ('MSE')
        plt.grid (True)

        plt.subplot (1, 2, 2)
        plt.plot (epochs, self.r2_scores, 'b-')
        plt.title ('R² Score Over Time')
        plt.xlabel ('Epoch')
        plt.ylabel ('R²')
        plt.grid (True)

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


class NaiveBayesVisualizer (Callback):
    def __init__(self, log_dir: str = "training_plots/naive_bayes"):
        super ().__init__ ()
        self.log_dir = log_dir
        os.makedirs (log_dir, exist_ok=True)
        self.predictions = []
        self.actuals = []
        self.class_probabilities = []
        self.accuracies = []
        self.class_wise_precision = []
        self.class_wise_recall = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        x, y = batch
        with torch.no_grad ():
            probs = pl_module (x)
            preds = torch.argmax (probs, dim=1)

            self.predictions.extend (preds.cpu ().numpy ())
            self.actuals.extend (y.cpu ().numpy ())
            self.class_probabilities.extend (probs.cpu ().numpy ())

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        self.accuracies.append (metrics.get ('accuracy', 0).item ())
        self.class_wise_precision.append (metrics.get ('precision', 0))
        self.class_wise_recall.append (metrics.get ('recall', 0))

        self._plot_metrics_over_time ()
        self._plot_class_probability_distributions ()
        self._plot_confusion_matrix ()

        self.predictions = []
        self.actuals = []
        self.class_probabilities = []

    def _plot_metrics_over_time(self):
        plt.figure (figsize=(10, 6))
        epochs = range (1, len (self.accuracies) + 1)
        plt.plot (epochs, self.accuracies, 'b-', label='Accuracy')

        plt.title ('Model Performance Over Time')
        plt.xlabel ('Epoch')
        plt.ylabel ('Score')
        plt.legend ()
        plt.grid (True)
        plt.savefig (os.path.join (self.log_dir, 'metrics_over_time.png'))
        plt.close ()

    def _plot_class_probability_distributions(self):
        if len (self.class_probabilities) > 0:
            probs = np.array (self.class_probabilities)
            n_classes = probs.shape [1]

            plt.figure (figsize=(12, 6))
            for i in range (n_classes):
                plt.hist (probs [:, i], bins=30, alpha=0.5, label=f'Class {i}')

            plt.title ('Class Probability Distributions')
            plt.xlabel ('Predicted Probability')
            plt.ylabel ('Count')
            plt.legend ()
            plt.grid (True)
            plt.savefig (os.path.join (self.log_dir, 'probability_distributions.png'))
            plt.close ()

    def _plot_confusion_matrix(self):
        if len (self.actuals) > 0:
            cm = confusion_matrix (self.actuals, self.predictions)
            plt.figure (figsize=(10, 8))
            sns.heatmap (cm, annot=True, fmt='d', cmap='Blues')
            plt.title ('Confusion Matrix')
            plt.xlabel ('Predicted Label')
            plt.ylabel ('True Label')
            plt.savefig (os.path.join (self.log_dir, 'confusion_matrix.png'))
            plt.close ()


class DecisionTreeVisualizer (Callback):
    def __init__(self, log_dir: str = "training_plots/decision_tree"):
        super ().__init__ ()
        self.log_dir = log_dir
        os.makedirs (log_dir, exist_ok=True)
        self.predictions = []
        self.actuals = []
        self.accuracies = []
        self.feature_importances = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        x, y = batch
        with torch.no_grad ():
            preds = pl_module (x)
            preds = torch.argmax (preds, dim=1)
            self.predictions.extend (preds.cpu ().numpy ())
            self.actuals.extend (y.cpu ().numpy ())

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self.accuracies.append (metrics.get ('accuracy', 0).item ())

        if pl_module.feature_importance is not None:
            self.feature_importances.append (pl_module.feature_importance.cpu ().numpy ())

        self._plot_accuracy_curve ()
        self._plot_feature_importance ()
        self._plot_decision_boundaries (trainer, pl_module)

        self.predictions = []
        self.actuals = []

    def _plot_accuracy_curve(self):
        plt.figure (figsize=(10, 6))
        epochs = range (1, len (self.accuracies) + 1)
        plt.plot (epochs, self.accuracies, 'g-')
        plt.title ('Decision Tree Accuracy Over Time')
        plt.xlabel ('Epoch')
        plt.ylabel ('Accuracy')
        plt.grid (True)
        plt.savefig (os.path.join (self.log_dir, 'accuracy_curve.png'))
        plt.close ()

    def _plot_feature_importance(self):
        if self.feature_importances:
            plt.figure (figsize=(12, 6))
            latest_importance = self.feature_importances [-1]
            indices = np.argsort (latest_importance) [::-1]

            plt.bar (range (len (latest_importance)), latest_importance [indices])
            plt.title ('Feature Importance')
            plt.xlabel ('Feature Index')
            plt.ylabel ('Importance Score')
            plt.tight_layout ()
            plt.savefig (os.path.join (self.log_dir, 'feature_importance.png'))
            plt.close ()

    def _plot_decision_boundaries(self, trainer, pl_module):
        if pl_module.n_features == 2:
            x_min, x_max = -3, 3
            y_min, y_max = -3, 3
            xx, yy = np.meshgrid (np.linspace (x_min, x_max, 100),
                                  np.linspace (y_min, y_max, 100))

            X_grid = torch.FloatTensor (np.c_ [xx.ravel (), yy.ravel ()]).to (pl_module.device)
            with torch.no_grad ():
                Z = pl_module (X_grid)
                Z = torch.argmax (Z, dim=1).cpu ().numpy ()

            Z = Z.reshape (xx.shape)

            plt.figure (figsize=(10, 8))
            plt.contourf (xx, yy, Z, alpha=0.4, cmap='viridis')
            plt.title ('Decision Boundaries')
            plt.xlabel ('Feature 1')
            plt.ylabel ('Feature 2')
            plt.colorbar ()
            plt.savefig (os.path.join (self.log_dir, 'decision_boundaries.png'))
            plt.close ()
