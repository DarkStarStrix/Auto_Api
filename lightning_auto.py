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


class AutoML (pl.LightningModule):
    """
    Automated Machine Learning Pipeline with PyTorch Lightning.
    Focuses on reliable optimizations and efficient training.
    """

    def __init__(self, config: Dict [str, Any], model: Optional [torch.nn.Module] = None):
        super ().__init__ ()
        self.steps_per_epoch = None
        self.config = config
        self.model = model or self._create_model ()
        self.save_hyperparameters (config)
        self._setup_pipeline ()

    def _create_model(self):
        """Create an improved model architecture."""
        input_dim = self.config ['model'] ['input_dim']
        hidden_dim = self.config ['model'] ['hidden_dim']
        output_dim = self.config ['model'] ['output_dim']

        if self.config ['model'] ['task'] == 'classification':
            return nn.Sequential (
                nn.Linear (input_dim, hidden_dim),
                nn.ReLU (),
                nn.Dropout (0.2),  # Add dropout for regularization
                nn.Linear (hidden_dim, hidden_dim),
                nn.ReLU (),
                nn.Dropout (0.2),
                nn.Linear (hidden_dim, output_dim)
            )
        else:
            return nn.Sequential (
                nn.Linear (input_dim, hidden_dim),
                nn.ReLU (),
                nn.Linear (hidden_dim, output_dim)
            )

    def _setup_pipeline(self):
        """Set up the automated pipeline components with improved logging."""
        print ("\nInitializing AutoML Pipeline...")
        print ("=" * 50)

        # Setup components with status logging
        print ("Setting up data engineering...")
        self._setup_data_engineering ()

        print ("Configuring optimizations...")
        self._setup_optimizations ()

        print ("Setting up training parameters...")
        self._setup_training ()

        print ("Pipeline initialization complete!")
        print ("=" * 50 + "\n")

    def _setup_data_engineering(self):
        """Set up data engineering components."""
        # Placeholder for data pipeline
        pass

    def _setup_optimizations(self):
        """Setup training optimizations with proper device handling."""
        opt_config = self.config.get ('optimization', {})

        # Set device type for autocast
        self.device_type = 'cuda' if torch.cuda.is_available () else 'cpu'

        # Automatic mixed precision with device check
        self.use_amp = opt_config.get ('mixed_precision', True)
        if self.use_amp and self.device_type == 'cpu':
            print ("Warning: Mixed precision requested but running on CPU. Disabling mixed precision.")
            self.use_amp = False

        # Gradient checkpointing with proper checks
        if opt_config.get ('gradient_checkpointing', False):
            if hasattr (self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable ()
                print ("Gradient checkpointing enabled.")
            else:
                print ("Warning: Model does not support gradient checkpointing. Skipping.")

        # Device strategy
        self.strategy = "auto"

        # Model compilation with version check
        self.compile_model = opt_config.get ('compile_model', False)
        if self.compile_model:
            if hasattr (torch, 'compile'):
                self.model = torch.compile (self.model)
                print ("Model compilation enabled.")
            else:
                print ("Warning: PyTorch version does not support model compilation. Skipping.")

    def _setup_training(self):
        """Set up training components."""
        train_config = self.config.get ('training', {})

        # Visualization and logging
        self.visualization = train_config.get ('visualization', True)
        self.metrics = train_config.get ('metrics', True)

        # Callbacks
        self.callbacks = []

        if self.visualization:
            self.callbacks.append (self.VisualizationCallback ())

        if self.metrics:
            self.callbacks.append (self.MetricsCallback ())

    def _setup_data_pipeline(self, train_data, val_data):
        """Initialize and set up the data pipeline when training starts."""
        data_config = self.config.get ('data', {})

        # Create data pipeline configuration
        self.data_pipeline = {
            'batch_size': data_config.get ('batch_size', 32),
            'num_workers': data_config.get ('num_workers', 7),
            'pin_memory': data_config.get ('pin_memory', True),
            'shuffle': data_config.get ('shuffle', True),
            'drop_last': data_config.get ('drop_last', False),
            'prefetch_factor': data_config.get ('prefetch_factor', 2),
        }

        # Process and prepare datasets
        self.train_dataset = self._prepare_dataset (train_data, is_train=True)
        self.val_dataset = self._prepare_dataset (val_data, is_train=False) if val_data is not None else None

    def _prepare_dataset(self, data, is_train: bool = True):
        """Prepare dataset with appropriate preprocessing."""
        if isinstance (data, (torch.Tensor, np.ndarray)):
            # Convert to torch dataset
            return _convert_to_dataset (data)
        elif isinstance (data, (DataLoader, TensorDataset)):
            # Already a PyTorch dataset or dataloader
            return data
        elif isinstance (data, pd.DataFrame):
            # Convert pandas DataFrame to dataset
            return self._convert_dataframe_to_dataset (data)
        else:
            raise ValueError (f"Unsupported data type: {type (data)}")

    def _convert_dataframe_to_dataset(self, df):
        """Convert pandas DataFrame to PyTorch Dataset."""
        # Get target column if specified in config
        target_col = self.config.get ('data', {}).get ('target_column')

        if target_col:
            features = df.drop (columns=[target_col])
            targets = df [target_col]
        else:
            features = df
            targets = np.zeros (len (df))  # Placeholder targets

        # Convert to tensors
        feature_tensor = torch.FloatTensor (features.values)
        target_tensor = torch.FloatTensor (targets.values)

        return TensorDataset (feature_tensor, target_tensor)

    def train_dataloader(self):
        """Create training DataLoader with optimized settings."""
        if not isinstance (self.train_dataset, DataLoader):
            # Calculate the optimal number of workers
            num_workers = min (os.cpu_count () - 1 or 1, 7)  # Leave one CPU core free

            return DataLoader (
                self.train_dataset,
                batch_size=self.data_pipeline ['batch_size'],
                num_workers=num_workers,  # Use calculated optimal workers
                pin_memory=True,  # Always True for better performance
                shuffle=True,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None
            )
        return self.train_dataset

    def val_dataloader(self):
        """Create validation DataLoader with optimized settings."""
        if self.val_dataset is not None and not isinstance (self.val_dataset, DataLoader):
            # Use the same number of workers as training
            num_workers = min (os.cpu_count () - 1 or 1, 7)

            return DataLoader (
                self.val_dataset,
                batch_size=self.data_pipeline ['batch_size'],
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None
            )
        return self.val_dataset

    def training_step(self, batch, batch_idx):
        """Training step with updated autocast."""
        # Use the new autocast API
        with torch.amp.autocast (device_type='cuda' if torch.cuda.is_available () else 'cpu',
                                 enabled=self.use_amp):
            loss = self._compute_loss (batch)

        # Log metrics
        self.log ('train_loss', loss, prog_bar=True)
        self.log ('lr', self.optimizers ().param_groups [0] ['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with updated autocast."""
        with torch.amp.autocast (device_type='cuda' if torch.cuda.is_available () else 'cpu',
                                 enabled=self.use_amp):
            loss = self._compute_loss (batch)

        self.log ('val_loss', loss, prog_bar=True)
        return loss

    def accumulate_grad_batches(self):
        """Calculate the number of steps to accumulate gradients."""
        return self.config.get ('optimization', {}).get ('accumulate_grad_batches', 1)

    def fit(self, train_data, val_data=None):
        """Initialize data pipeline and start training."""
        # Setup data pipeline when training starts
        self._setup_data_pipeline (train_data, val_data)

        # Calculate steps per epoch for schedulers
        self.steps_per_epoch = len (self.train_dataset) // self.data_pipeline ['batch_size']

        # Create trainer with optimized settings
        trainer = pl.Trainer (
            max_epochs=self.config ['training'].get ('epochs', 10),
            accelerator="gpu" if torch.cuda.is_available () else "cpu",
            devices="auto",
            precision="16-mixed" if self.use_amp else "32",
            strategy="auto",
            accumulate_grad_batches=self.accumulate_grad_batches (),
            gradient_clip_val=self.config ['training'].get ('gradient_clip_val', 1.0),
            callbacks=self._get_callbacks (),
            logger=self._get_logger (),
            log_every_n_steps=10  # Set log_every_n_steps to 10
        )

        # Start training
        trainer.fit (self)

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

        # Add visualization callback

        # Add metrics callback

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

    class VisualizationCallback (Callback):
        """Custom callback for visualizations during training."""

        def __init__(self, log_dir: str = "training_plots"):
            super ().__init__ ()
            self.log_dir = log_dir
            self.training_losses = []
            self.validation_losses = []
            self.learning_rates = []
            os.makedirs (log_dir, exist_ok=True)

        def on_train_epoch_end(self, trainer, pl_module):
            """Create and save visualizations after each epoch."""
            # Collect metrics
            self.training_losses.append (trainer.callback_metrics.get ('train_loss', 0))
            self.validation_losses.append (trainer.callback_metrics.get ('val_loss', 0))
            self.learning_rates.append (trainer.optimizers [0].param_groups [0] ['lr'])

            # Create plots
            self._plot_losses ()
            self._plot_lr_curve ()
            if hasattr (pl_module, 'model'):
                self._plot_gradients (pl_module.model)
                self._plot_model_predictions (pl_module)

        def _plot_losses(self):
            """Plot training and validation losses."""
            plt.figure (figsize=(10, 6))
            plt.plot (self.training_losses, label='Training Loss', marker='o')
            plt.plot (self.validation_losses, label='Validation Loss', marker='o')
            plt.title ('Loss During Training')
            plt.xlabel ('Epoch')
            plt.ylabel ('Loss')
            plt.legend ()
            plt.grid (True)
            plt.savefig (os.path.join (self.log_dir, 'loss_plot.png'))
            plt.close ()

        def _plot_lr_curve(self):
            """Plot learning rate curve."""
            plt.figure (figsize=(10, 6))
            plt.plot (self.learning_rates, marker='o')
            plt.title ('Learning Rate Schedule')
            plt.xlabel ('Epoch')
            plt.ylabel ('Learning Rate')
            plt.yscale ('log')
            plt.grid (True)
            plt.savefig (os.path.join (self.log_dir, 'lr_curve.png'))
            plt.close ()

        def _plot_gradients(self, model):
            """Plot gradient distributions."""
            gradients = []
            for param in model.parameters ():
                if param.grad is not None:
                    gradients.extend (param.grad.cpu ().numpy ().flatten ())

            if gradients:
                plt.figure (figsize=(10, 6))
                sns.histplot (gradients, bins=50)
                plt.title ('Gradient Distribution')
                plt.xlabel ('Gradient Value')
                plt.ylabel ('Count')
                plt.savefig (os.path.join (self.log_dir, 'gradient_dist.png'))
                plt.close ()

        def _plot_model_predictions(self, pl_module):
            """Plot model predictions for classification with proper visualization."""
            try:
                # Get model configuration
                num_samples = 100
                input_dim = pl_module.config ['model'] ['input_dim']
                output_dim = pl_module.config ['model'] ['output_dim']

                # Create structured example data
                example_data = torch.randn (num_samples, input_dim)

                # Get model predictions
                with torch.no_grad ():
                    logits = pl_module.model (example_data)
                    predictions = logits.argmax (dim=1).cpu ().numpy ()

                # Create actual labels (for visualization purposes)
                # Instead of random, create a structured distribution
                actual_values = np.array ([i % output_dim for i in range (num_samples)])

                # Create confusion matrix data
                confusion_data = {
                    'actual': [],
                    'predicted': [],
                    'count': []
                }

                for i in range (output_dim):
                    for j in range (output_dim):
                        mask = (actual_values == i) & (predictions == j)
                        count = np.sum (mask)
                        confusion_data ['actual'].append (i)
                        confusion_data ['predicted'].append (j)
                        confusion_data ['count'].append (count)

                # Create the plot
                plt.figure (figsize=(10, 8))

                # Plot accuracy heatmap
                conf_matrix = np.zeros ((output_dim, output_dim))
                for a, p, c in zip (confusion_data ['actual'],
                                    confusion_data ['predicted'],
                                    confusion_data ['count']):
                    conf_matrix [a, p] = c

                plt.imshow (conf_matrix, cmap='Blues')
                plt.colorbar (label='Count')

                # Add labels
                plt.title ('Classification Results: Predicted vs Actual Classes')
                plt.xlabel ('Predicted Class')
                plt.ylabel ('Actual Class')

                # Add grid
                plt.grid (False)

                # Add value annotations
                for i in range (output_dim):
                    for j in range (output_dim):
                        plt.text (j, i, f'{int (conf_matrix [i, j])}',
                                  ha='center', va='center')

                # Set ticks
                plt.xticks (range (output_dim), [f'Class {i}' for i in range (output_dim)])
                plt.yticks (range (output_dim), [f'Class {i}' for i in range (output_dim)])

                # Adjust layout and save
                plt.tight_layout ()
                plt.savefig (os.path.join (self.log_dir, 'classification_results.png'))
                plt.close ()

                # Additional plot: Class distribution
                plt.figure (figsize=(10, 6))
                unique, counts = np.unique (predictions, return_counts=True)
                plt.bar (unique, counts)
                plt.title ('Distribution of Predicted Classes')
                plt.xlabel ('Class')
                plt.ylabel ('Count')
                plt.xticks (unique, [f'Class {i}' for i in unique])
                plt.grid (True, axis='y')
                plt.savefig (os.path.join (self.log_dir, 'class_distribution.png'))
                plt.close ()

            except Exception as e:
                print (f"Warning: Could not plot model predictions. Error: {str (e)}")
                import traceback
                traceback.print_exc ()

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
