import pytorch_lightning as pl
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel (pl.LightningModule, ABC):
    def __init__(self, config: Dict [str, Any]):
        super ().__init__ ()
        self.config = config
        self.save_hyperparameters (config)
        self.model = self._create_model ()

    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def _compute_loss(self, batch):
        pass

    def forward(self, x):
        return self.model (x)

    def configure_optimizers(self):
        return torch.optim.Adam (self.parameters (), lr=0.001)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss (batch)
        self.log ('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss (batch)
        self.log ('val_loss', loss)
        return loss
