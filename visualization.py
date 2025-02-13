# visualization.py
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pydot
import seaborn as sns
import torch
from matplotlib.colors import LogNorm
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz


class LinearRegressionVisualizer(Callback):
    """Visualization callback for linear regression."""

    def __init__(self, log_dir: str = "training_plots/linear_regression", feature_names: list = None):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.predictions = []
        self.actuals = []
        self.mse_scores = []
        self.r2_scores = []
        self.training_losses = []
        self.validation_losses = []
        self.residuals = []
        self.feature_names = feature_names if feature_names else ["Feature", "Target"]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        """Collect predictions and actual values."""
        x, y = batch
        with torch.no_grad():
            preds = pl_module(x)
            self.predictions.extend(preds.cpu().numpy().flatten())
            self.actuals.extend(y.cpu().numpy().flatten())
            self.residuals.extend((preds - y).cpu().numpy().flatten())

    def on_train_epoch_end(self, trainer, pl_module):
        """Plot visualizations at the end of each epoch."""
        metrics = trainer.callback_metrics

        # Collect metrics
        self.mse_scores.append(metrics.get('mse', 0).item())
        self.r2_scores.append(metrics.get('r2_score', 0).item())
        self.training_losses.append(metrics.get('train_loss', 0).item())
        self.validation_losses.append(metrics.get('val_loss', 0).item())

        # Create plots
        self._plot_metrics_over_time()
        self._plot_residuals()
        self._plot_predictions_vs_actual()
        self._plot_residual_distribution()

        # Clear batch data
        self.predictions = []
        self.actuals = []
        self.residuals = []

    def _plot_metrics_over_time(self):
        """Plot MSE and R² score over time."""
        plt.figure(figsize=(15, 5))
        epochs = range(1, len(self.mse_scores) + 1)

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.mse_scores, 'r-')
        plt.title('Mean Squared Error Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.r2_scores, 'b-')
        plt.title('R² Score Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'metrics_over_time.png'))
        plt.close()

    def _plot_residuals(self):
        """Plot residuals over time."""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.residuals) + 1)
        plt.plot(epochs, self.residuals, 'g-')
        plt.title('Residuals Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Residual Value')
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'residuals.png'))
        plt.close()

    def _plot_predictions_vs_actual(self):
        """Plot predicted vs actual values."""
        if len (self.predictions) > 0:
            plt.figure (figsize=(10, 6))
            plt.scatter (self.actuals, self.predictions, alpha=0.5)
            self.feature_names = self.feature_names if self.feature_names else ["Actual Value", "Predicted Value"]

            min_val = min (min (self.actuals), min (self.predictions))
            max_val = max (max (self.actuals), max (self.predictions))
            plt.plot ([min_val, max_val], [min_val, max_val], 'r--')

            plt.title ('Predictions vs Actual Values')
            plt.xlabel (self.feature_names [0])
            plt.ylabel (self.feature_names [1])

            plt.grid (True)
            plt.savefig (os.path.join (self.log_dir, 'predictions_vs_actual.png'))
            plt.close ()

    def _plot_residual_distribution(self):
        """Plot distribution of residuals."""
        if len(self.residuals) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(self.residuals, bins=50, edgecolor='black')
            plt.title('Distribution of Residuals')
            plt.xlabel('Residual Value')
            plt.ylabel('Count')
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, 'residual_distribution.png'))
            plt.close()


class LogisticRegressionVisualizer(Callback):
    """Visualization callback for logistic regression."""

    def __init__(self, log_dir: str = "training_plots/logistic_regression", feature_names: list = None):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.predictions = []
        self.actuals = []
        self.probabilities = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.training_losses = []
        self.validation_losses = []
        self.feature_names = feature_names if feature_names else ["Feature", "Target"]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        """Collect predictions and actual values."""
        x, y = batch
        with torch.no_grad():
            probs = pl_module(x)
            preds = (probs > 0.5).float()

            self.predictions.extend(preds.cpu().numpy().flatten())
            self.actuals.extend(y.cpu().numpy().flatten())
            self.probabilities.extend(probs.cpu().numpy().flatten())

    def on_train_epoch_end(self, trainer, pl_module):
        """Plot visualizations at the end of each epoch."""
        metrics = trainer.callback_metrics

        # Collect metrics
        self.accuracies.append(metrics.get('accuracy', 0).item())
        self.precisions.append(metrics.get('precision', 0).item())
        self.recalls.append(metrics.get('recall', 0).item())
        self.training_losses.append(metrics.get('train_loss', 0).item())
        self.validation_losses.append(metrics.get('val_loss', 0).item())

        # Create plots
        self._plot_metrics_over_time()
        self._plot_probability_distribution()
        self._plot_custom_confusion_matrix()

        # Clear batch data
        self.predictions = []
        self.actuals = []
        self.probabilities = []

    def _plot_metrics_over_time(self):
        """Plot accuracy, precision, and recall over time."""
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(self.accuracies) + 1)

        plt.plot(epochs, self.accuracies, 'b-', label='Accuracy')
        plt.plot(epochs, self.precisions, 'g-', label='Precision')
        plt.plot(epochs, self.recalls, 'r-', label='Recall')

        plt.title('Classification Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'metrics_over_time.png'))
        plt.close()

    def _plot_probability_distribution(self):
        """Plot distribution of predicted probabilities."""
        if len(self.probabilities) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(self.probabilities, bins=50, edgecolor='black')
            plt.title('Distribution of Predicted Probabilities')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Count')
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, 'probability_distribution.png'))
            plt.close()

    def _plot_custom_confusion_matrix(self):
        """Plot simplified confusion matrix."""
        if len(self.actuals) > 0:
            # Calculate confusion matrix manually
            predictions = np.array(self.predictions)
            actuals = np.array(self.actuals)

            tp = np.sum((predictions == 1) & (actuals == 1))
            tn = np.sum((predictions == 0) & (actuals == 0))
            fp = np.sum((predictions == 1) & (actuals == 0))
            fn = np.sum((predictions == 0) & (actuals == 1))

            cm = np.array([[tn, fp], [fn, tp]])

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.colorbar()

            classes = ['Negative', 'Positive']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)

            # Add text annotations
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black")

            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'confusion_matrix.png'))
            plt.close()


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
        self._plot_decision_boundaries (pl_module)

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

    def _plot_decision_boundaries(self, pl_module):
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


class KMeansVisualizer (Callback):
    def __init__(self, log_dir: str = "training_plots/kmeans"):
        super ().__init__ ()
        try:
            # Ensure log directory exists
            self.log_dir = log_dir
            os.makedirs (log_dir, exist_ok=True)
            print (f"Initialized KMeansVisualizer with log directory: {log_dir}")

            # Initialize tracking lists
            self.predictions = []
            self.data_points = []
            self.inertias = []
            self.cluster_sizes_history = []
            self.silhouette_scores = []

            # Initialize plot figures
            self.metrics_fig, self.metrics_axs = plt.subplots (2, 2, figsize=(15, 10))
            self.cluster_fig = plt.figure (figsize=(10, 8))
            self.cluster_ax = self.cluster_fig.add_subplot (111)

            # Close initial test figures
            plt.close (self.metrics_fig)
            plt.close (self.cluster_fig)
        except Exception as e:
            print (f"Error initializing KMeansVisualizer: {str (e)}")
            raise

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        try:
            x, _ = batch
            with torch.no_grad ():
                distances, assignments = pl_module (x)
                self.predictions.extend (assignments.cpu ().numpy ().tolist ())
                self.data_points.extend (x.cpu ().numpy ().tolist ())
        except Exception as e:
            print (f"Error in validation_batch_end: {str (e)}")

    def on_train_epoch_end(self, trainer, pl_module):
        try:
            # Get metrics
            metrics = trainer.callback_metrics

            # Store inertia
            inertia = metrics.get ('inertia', 0)
            if isinstance (inertia, torch.Tensor):
                inertia = inertia.item ()
            self.inertias.append (inertia)

            # Store cluster sizes
            cluster_sizes = {}
            for i in range (pl_module.n_clusters):
                size_key = f'cluster_size_{i}'
                size = metrics.get (size_key, 0)
                if isinstance (size, torch.Tensor):
                    size = size.item ()
                cluster_sizes [f'cluster_{i}'] = size
            self.cluster_sizes_history.append (cluster_sizes)

            # Update plots
            self._update_metric_plots (trainer.current_epoch)
            self._update_cluster_plot (pl_module)

            # Clear batch data
            self.predictions.clear ()
            self.data_points.clear ()

        except Exception as e:
            print (f"Error in train_epoch_end: {str (e)}")
            import traceback
            traceback.print_exc ()

    def _update_metric_plots(self, epoch):
        try:
            # Clear previous plots
            for ax in self.metrics_axs.flat:
                ax.clear ()

            # Plot 1: Inertia over time
            ax = self.metrics_axs [0, 0]
            epochs = range (len (self.inertias))
            ax.plot (epochs, self.inertias, 'b-', label='Inertia')
            ax.set_title ('Inertia over Epochs')
            ax.set_xlabel ('Epoch')
            ax.set_ylabel ('Inertia')
            ax.grid (True)
            ax.legend ()

            # Plot 2: Cluster sizes over time
            ax = self.metrics_axs [0, 1]
            if self.cluster_sizes_history:
                for cluster in range (len (self.cluster_sizes_history [0])):
                    sizes = [epoch_sizes [f'cluster_{cluster}']
                             for epoch_sizes in self.cluster_sizes_history]
                    ax.plot (epochs, sizes, label=f'Cluster {cluster}')
                ax.set_title ('Cluster Sizes over Epochs')
                ax.set_xlabel ('Epoch')
                ax.set_ylabel ('Proportion of Points')
                ax.grid (True)
                ax.legend ()

            # Plot 3: Inertia change rate
            ax = self.metrics_axs [1, 0]
            if len (self.inertias) > 1:
                changes = np.diff (self.inertias)
                ax.plot (epochs [1:], changes, 'r-', label='Change')
                ax.set_title ('Inertia Change Rate')
                ax.set_xlabel ('Epoch')
                ax.set_ylabel ('Change')
                ax.grid (True)
                ax.legend ()

            # Plot 4: Current metrics summary
            ax = self.metrics_axs [1, 1]
            ax.axis ('off')
            status_text = f"Epoch: {epoch}\n"
            status_text += f"Current Inertia: {self.inertias [-1]:.4f}\n"
            if len (self.inertias) > 1:
                status_text += f"Change: {self.inertias [-1] - self.inertias [-2]:.4f}"
            ax.text (0.1, 0.5, status_text, fontsize=10)

            self.metrics_fig.suptitle ('K-Means Training Metrics', fontsize=14)
            self.metrics_fig.tight_layout ()
            self.metrics_fig.savefig (os.path.join (self.log_dir, 'kmeans_metrics.png'))

        except Exception as e:
            print (f"Error in _update_metric_plots: {str (e)}")
            traceback.print_exc ()

    def _update_cluster_plot(self, pl_module):
        try:
            # Create a simple 2D grid
            grid_size = 100
            x = np.linspace (-4, 4, grid_size)
            y = np.linspace (-4, 4, grid_size)
            X, Y = np.meshgrid (x, y)

            # Clear current plot
            self.cluster_ax.clear ()

            # Get centroids (we'll only plot first 2 dimensions)
            centroids = pl_module.centroids.detach ().cpu ().numpy () [:, :2]

            # For each point in the grid, find the nearest centroid
            XX = X.flatten ()
            YY = Y.flatten ()
            points = np.column_stack ([XX, YY])

            # Calculate distances to centroids
            assignments = []
            for point in points:
                dists = np.linalg.norm (centroids - point, axis=1)
                assignments.append (np.argmin (dists))

            # Reshape assignments back to grid
            assignments = np.array (assignments).reshape (grid_size, grid_size)

            # Plot the grid coloring
            self.cluster_ax.contourf (X, Y, assignments, alpha=0.3, cmap='viridis')

            # Plot centroids
            self.cluster_ax.scatter (
                centroids [:, 0], centroids [:, 1],
                c='red', marker='x', s=200, linewidth=3,
                label='Centroids'
            )

            # Set labels and title
            self.cluster_ax.set_xlabel ('Dimension 1')
            self.cluster_ax.set_ylabel ('Dimension 2')
            self.cluster_ax.set_title ('K-Means Clusters (2D Projection)')
            self.cluster_ax.legend ()

            # Save plot
            self.cluster_fig.savefig (
                os.path.join (self.log_dir, 'kmeans_clusters.png'),
                bbox_inches='tight'
            )

        except Exception as e:
            print (f"Error in cluster plot: {e}")


class LightGBMVisualizer (Callback):
    def __init__(self, log_dir: str = "training_plots/lightgbm"):
        super ().__init__ ()
        self.log_dir = log_dir
        os.makedirs (log_dir, exist_ok=True)

        # Initialize metrics history
        self.metrics_history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'feature_importance': []
        }
        self.predictions_history = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        x, y = batch
        with torch.no_grad ():
            preds = pl_module (x)
            if isinstance (preds, torch.Tensor):
                preds = preds.cpu ().numpy ()
            self.predictions_history.extend (preds)

    def on_train_epoch_end(self, trainer, pl_module):
        try:
            metrics = trainer.callback_metrics

            # Store metrics
            train_acc = metrics.get ('train_accuracy', 0)
            val_acc = metrics.get ('val_accuracy', 0)

            # Convert from tensor if necessary
            if isinstance (train_acc, torch.Tensor):
                train_acc = train_acc.item ()
            if isinstance (val_acc, torch.Tensor):
                val_acc = val_acc.item ()

            self.metrics_history ['train_accuracy'].append (train_acc)
            self.metrics_history ['val_accuracy'].append (val_acc)

            # Store feature importance if available
            if hasattr (pl_module.model, 'feature_importance'):
                importance = pl_module.model.feature_importance ()
                if importance is not None:
                    importance = importance / importance.sum ()
                    self.metrics_history ['feature_importance'].append (importance)

            # Create plots
            self._plot_metrics ()
            if self.metrics_history ['feature_importance']:
                self._plot_feature_importance ()

            self.predictions_history = []

        except Exception as e:
            print (f"Error in visualization: {str (e)}")
            import traceback
            traceback.print_exc ()

    def _plot_metrics(self):
        plt.figure (figsize=(12, 5))
        epochs = range (1, len (self.metrics_history ['train_accuracy']) + 1)

        # Plot accuracy
        plt.plot (epochs, self.metrics_history ['train_accuracy'], 'b-', label='Train Accuracy')
        plt.plot (epochs, self.metrics_history ['val_accuracy'], 'r-', label='Validation Accuracy')
        plt.title ('Model Accuracy')
        plt.xlabel ('Epoch')
        plt.ylabel ('Accuracy')
        plt.grid (True)
        plt.legend ()

        plt.tight_layout ()
        plt.savefig (os.path.join (self.log_dir, 'accuracy.png'))
        plt.close ()

    def _plot_feature_importance(self):
        if not self.metrics_history ['feature_importance']:
            return

        plt.figure (figsize=(10, 6))
        importance = self.metrics_history ['feature_importance'] [-1]  # Get latest
        feature_indices = range (len (importance))

        plt.bar (feature_indices, importance)
        plt.title ('Feature Importance (LightGBM)')
        plt.xlabel ('Feature Index')
        plt.ylabel ('Importance Score')

        # Add value labels on top of bars
        for i, v in enumerate (importance):
            plt.text (i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout ()
        plt.savefig (os.path.join (self.log_dir, 'feature_importance.png'))
        plt.close ()


class RandomForestVisualizer (Callback):
    def __init__(self, log_dir: str = "Training_plots/random_forest"):
        super ().__init__ ()
        self.log_dir = log_dir
        os.makedirs (log_dir, exist_ok=True)

        # Initialize metrics history
        self.metrics_history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'feature_importance': [],
            'tree': []
        }
        self.predictions_history = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        x, y = batch
        with torch.no_grad ():
            preds = pl_module (x)
            if isinstance (preds, torch.Tensor):
                preds = preds.cpu ().numpy ()
            self.predictions_history.extend (preds)

    def on_train_epoch_end(self, trainer, pl_module):
        try:
            metrics = trainer.callback_metrics

            # Store metrics
            train_acc = metrics.get ('train_accuracy', 0)
            val_acc = metrics.get ('val_accuracy', 0)

            # Convert from tensor if necessary
            if isinstance (train_acc, torch.Tensor):
                train_acc = train_acc.item ()
            if isinstance (val_acc, torch.Tensor):
                val_acc = val_acc.item ()

            self.metrics_history ['train_accuracy'].append (train_acc)
            self.metrics_history ['val_accuracy'].append (val_acc)

            # Store feature importance if available
            if hasattr (pl_module.model, 'feature_importance'):
                importance = pl_module.model.feature_importance ()
                if importance is not None:
                    importance = importance / importance.sum ()
                    self.metrics_history ['feature_importance'].append (importance)

            # Store tree if available
            if hasattr (pl_module.model, 'estimators_'):
                for i, estimator in enumerate (pl_module.model.estimators_):
                    self._plot_tree (estimator, feature_names=pl_module.feature_names)

            # Create plots
            self._plot_metrics ()
            if self.metrics_history ['feature_importance']:
                self._plot_feature_importance ()

            self.predictions_history = []

        except Exception as e:
            print (f"Error in visualization: {str (e)}")
            import traceback
            traceback.print_exc ()

    def _plot_metrics(self):
        plt.figure (figsize=(12, 5))
        epochs = range (1, len (self.metrics_history ['train_accuracy']) + 1)

        # Plot accuracy
        plt.plot (epochs, self.metrics_history ['train_accuracy'], 'b-', label='Train Accuracy')
        plt.plot (epochs, self.metrics_history ['val_accuracy'], 'r-', label='Validation Accuracy')
        plt.title ('Model Accuracy')
        plt.xlabel ('Epoch')
        plt.ylabel ('Accuracy')
        plt.grid (True)
        plt.legend ()

        plt.tight_layout ()
        plt.savefig (os.path.join (self.log_dir, 'accuracy.png'))
        plt.close ()

    def _plot_feature_importance(self):
        if not self.metrics_history ['feature_importance']:
            return

        plt.figure (figsize=(10, 6))
        importance = self.metrics_history ['feature_importance'] [-1]

        feature_indices = range (len (importance))
        plt.bar (feature_indices, importance)

        plt.title ('Feature Importance (Random Forest)')
        plt.xlabel ('Feature Index')
        plt.ylabel ('Importance Score')

        # Add value labels on top of bars
        for i, v in enumerate (importance):
            plt.text (i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout ()
        plt.savefig (os.path.join (self.log_dir, 'feature_importance.png'))
        plt.close ()

    def _plot_tree(self, estimator, feature_names):
        try:
            dot_data = export_graphviz (estimator, out_file=None, feature_names=feature_names, filled=True,
                                        rounded=True, special_characters=True)
            graphs = pydot.graph_from_dot_data (dot_data)
            if graphs:
                graph = graphs [0]  # Access the first graph in the list
                graph.write_png (os.path.join (self.log_dir, f'tree_{len (self.metrics_history ["tree"])}.png'))
        except Exception as e:
            print (f"Error in _plot_tree: {str (e)}")
            raise


class SVMVisualizer (Callback):
    def __init__(self, log_dir: str = "training_plots/svm"):
        super ().__init__ ()
        self.log_dir = log_dir
        os.makedirs (log_dir, exist_ok=True)
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'decision_boundary': []
        }

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        try:
            metrics = trainer.callback_metrics
            train_loss = metrics.get ('train_loss', 0)
            val_loss = metrics.get ('val_loss', 0)

            if isinstance (train_loss, torch.Tensor):
                train_loss = train_loss.item ()
            if isinstance (val_loss, torch.Tensor):
                val_loss = val_loss.item ()

            self.metrics_history ['train_loss'].append (train_loss)
            self.metrics_history ['val_loss'].append (val_loss)

            self._plot_metrics ()

            # Plot decision boundary if input dimension is 2
            if pl_module.input_dim == 2:
                X = pl_module.validation_data  # Ensure this is the correct data
                self.plot_density_function (pl_module.model, X)

        except Exception as e:
            print (f"Error in visualization: {str (e)}")
            import traceback
            traceback.print_exc ()

    def _plot_metrics(self):
        plt.figure (figsize=(12, 5))
        epochs = range (1, len (self.metrics_history ['train_loss']) + 1)

        plt.plot (epochs, self.metrics_history ['train_loss'], 'b-', label='Train Loss')
        plt.plot (epochs, self.metrics_history ['val_loss'], 'r-', label='Validation Loss')
        plt.title ('Model Loss')
        plt.xlabel ('Epoch')
        plt.ylabel ('Loss')
        plt.grid (True)
        plt.legend ()

        plt.tight_layout ()
        plt.savefig (os.path.join (self.log_dir, 'loss.png'))
        plt.close ()

    @staticmethod
    def plot_linear_decision_boundary(model, X):
        model.coef_ [0]
        model.intercept_ [0]

        x_min, x_max = X [:, 0].min () - 1, X [:, 0].max () + 1
        y_min, y_max = X [:, 1].min () - 1, X [:, 1].max () + 1
        xx, yy = np.meshgrid (np.arange (x_min, x_max, 0.01),
                              np.arange (y_min, y_max, 0.01))

        Z = model.predict (np.c_ [xx.ravel (), yy.ravel ()])
        Z = Z.reshape (xx.shape)

        plt.contourf (xx, yy, Z, alpha=0.8)
        plt.scatter (X [:, 0], X [:, 1], c='black', edgecolor='k', s=20)
        plt.xlim (xx.min (), xx.max ())
        plt.ylim (yy.min (), yy.max ())
        plt.title ("Decision Boundary")
        plt.show ()

    def plot_density_function(self, model, X):
        pass


class GaussianMixtureVisualizer(Callback):
    def __init__(self, log_dir: str = "training_plots/gaussian_mixture"):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'decision_boundary': []
        }

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        try:
            metrics = trainer.callback_metrics
            train_loss = metrics.get ('train_loss', 0)
            val_loss = metrics.get ('val_loss', 0)

            if isinstance (train_loss, torch.Tensor):
                train_loss = train_loss.item ()
            if isinstance (val_loss, torch.Tensor):
                val_loss = val_loss.item ()

            self.metrics_history ['train_loss'].append (train_loss)
            self.metrics_history ['val_loss'].append (val_loss)

            self._plot_metrics ()

            # Plot decision boundary if input dimension is 2
            if pl_module.input_dim == 2:
                X_train = pl_module.validation_data
                self.plot_gmm_density (X_train, pl_module.model, self.log_dir)

        except Exception as e:
            print (f"Error in visualization: {str (e)}")
            import traceback
            traceback.print_exc ()

    def _plot_metrics(self):
        plt.figure(figsize=(12, 5))
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)

        plt.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss')
        plt.plot(epochs, self.metrics_history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'loss.png'))
        plt.close()

    @staticmethod
    def plot_gmm_density(X_train, clf, log_dir):
        x = np.linspace (-20.0, 30.0)
        y = np.linspace (-20.0, 40.0)
        X, Y = np.meshgrid (x, y)
        XX = np.array ([X.ravel (), Y.ravel ()]).T
        Z = -clf.score_samples (XX)
        Z = Z.reshape (X.shape)

        plt.contour (X, Y, Z, norm=LogNorm (vmin=1.0, vmax=1000.0), levels=np.logspace (0, 3, 10))
        plt.colorbar (shrink=0.8, extend="both")
        plt.scatter (X_train [:, 0], X_train [:, 1], 0.8)

        plt.title ("Negative log-likelihood predicted by a GMM")
        plt.axis ("tight")
        plt.savefig (os.path.join (log_dir, 'gmm_density.png'))
        plt.close ()
