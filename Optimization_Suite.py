from typing import Dict, Any, Optional

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.quantization import quantize_dynamic


def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Dynamic quantization of the model"""
    return quantize_dynamic (
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )


class ModelOptimizer:
    def __init__(self, model_class, data_loader, config):
        self.model_class = model_class
        self.data_loader = data_loader
        self.base_config = config
        self.best_trial = None

    def optimize_hyperparameters(self, n_trials: int = 20) -> Dict [str, Any]:
        study = optuna.create_study (direction="minimize")
        study.optimize (self._objective, n_trials=n_trials)
        self.best_trial = study.best_trial
        return study.best_params

    def _objective(self, trial) -> float:
        config = self.base_config.copy ()
        config.update ({
            'learning_rate': trial.suggest_float ('learning_rate', 1e-5, 1e-1, log=True),
            'batch_size': trial.suggest_int ('batch_size', 16, 128, step=16),
            'optimizer': trial.suggest_categorical ('optimizer', ['adam', 'sgd', 'adamw']),
            'weight_decay': trial.suggest_float ('weight_decay', 1e-5, 1e-2, log=True)
        })

        model = self.model_class (config)
        trainer = pl.Trainer (
            max_epochs=5,
            callbacks=[EarlyStopping (monitor='val_loss', patience=3)],
            enable_progress_bar=False
        )

        trainer.fit (model, self.data_loader)
        return trainer.callback_metrics ['val_loss'].item ()

    def get_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        if self.best_trial is None:
            raise ValueError ("Run optimize_hyperparameters first")

        opt_name = self.best_trial.params ['optimizer']
        lr = self.best_trial.params ['learning_rate']
        weight_decay = self.best_trial.params ['weight_decay']

        optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'adamw': torch.optim.AdamW
        }

        return optimizers [opt_name] (
            model.parameters (),
            lr=lr,
            weight_decay=weight_decay
        )

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional [torch.optim.lr_scheduler._LRScheduler]:
        scheduler_type = self.base_config.get ('scheduler', 'cosine')

        if scheduler_type == 'plateau':
            return ReduceLROnPlateau (
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR (
                optimizer,
                T_max=self.base_config.get ('epochs', 100),
                eta_min=1e-6
            )
        return None

    @staticmethod
    def apply_pruning(model: torch.nn.Module, amount: float = 0.3) -> torch.nn.Module:
        """Prune model weights"""
        for name, module in model.named_modules ():
            if isinstance (module, torch.nn.Linear):
                torch.nn.utils.prune.l1_unstructured (
                    module,
                    name='weight',
                    amount=amount
                )
        return model


class MemoryTracker:
    @staticmethod
    def get_model_size(model: torch.nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters ():
            param_size += param.nelement () * param.element_size ()
        buffer_size = 0
        for buffer in model.buffers ():
            buffer_size += buffer.nelement () * buffer.element_size ()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb

    @staticmethod
    def log_memory_stats(model: torch.nn.Module, phase: str = ""):
        """Log memory statistics"""
        if torch.cuda.is_available ():
            print (f"\n{phase} Memory Stats:")
            print (f"Allocated: {torch.cuda.memory_allocated () / 1024 ** 2:.2f}MB")
            print (f"Cached: {torch.cuda.memory_reserved () / 1024 ** 2:.2f}MB")
        print (f"Model Size: {MemoryTracker.get_model_size (model):.2f}MB")
