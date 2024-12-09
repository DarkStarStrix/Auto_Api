import os
import neptune
from neptune.types import File
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import torch


class NeptuneModelTracker (Callback):
    def __init__(self, api_token: str, project: str):
        super ().__init__ ()
        self.run = neptune.init_run (
            project=project,
            api_token=api_token,
            capture_hardware_metrics=True
        )
        self.epoch_losses = []
        self.val_metrics = {}

    def on_train_start(self, trainer, pl_module):
        self.run ["config"] = pl_module.config
        self.run ["model/parameters"] = sum (p.numel () for p in pl_module.parameters ())
        self.run ["model/architecture"] = str (pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Log metrics
        self.run ["metrics/train/loss"].append (metrics.get ("train_loss", 0))
        self.run ["metrics/val/loss"].append (metrics.get ("val_loss", 0))

        if "accuracy" in metrics:
            self.run ["metrics/val/accuracy"].append (metrics ["accuracy"])

        # Generate and log confusion matrix
        if hasattr (pl_module, "predictions") and hasattr (pl_module, "targets"):
            fig = plt.figure (figsize=(8, 8))
            sns.heatmap (pl_module.confusion_matrix (), annot=True, fmt="d")
            plt.title ("Confusion Matrix")
            self.run ["visualizations/confusion_matrix"].upload (File.as_image (fig))
            plt.close ()

    def on_train_end(self, trainer, pl_module):
        # Save final model metrics
        self.run ["final_metrics"] = {
            "best_val_loss": min (self.val_metrics.get ("val_loss", [float ('inf')]))
        }

        # Save model artifacts
        model_path = "model.pt"
        torch.save (pl_module.state_dict (), model_path)
        self.run ["model/checkpoints"].upload (model_path)
        os.remove (model_path)

        self.run.stop ()

    def __del__(self):
        if hasattr (self, 'run'):
            self.run.stop ()


# Example usage with AutoML
class AutoML:
    def __init__(self, config: Dict [str, Any]):
        self.config = config
        self.model = self._create_model ()

        # Initialize Neptune tracking
        self.neptune_tracker = NeptuneModelTracker (
            api_token=os.getenv ("NEPTUNE_API_TOKEN"),
            project=os.getenv ("NEPTUNE_PROJECT")
        )

    def fit(self, train_data, val_data=None):
        trainer = pl.Trainer (
            max_epochs=self.config.get ('epochs', 10),
            callbacks=[self.neptune_tracker],
            enable_progress_bar=True
        )

        trainer.fit (self.model, train_data, val_data)
        return self.model
