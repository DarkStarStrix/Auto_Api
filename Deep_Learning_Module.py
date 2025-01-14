import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from Model_Types import load_model
from Model_Library import get_deep_learning_config

class DeepLearningModule(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        self.model = load_model(model_config)
        self.train_data = None  # Placeholder for training data
        self.val_data = None  # Placeholder for validation data

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.model.training_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def train_dataloader(self):
        if self.train_data is None:
            raise ValueError("Training data not set. Please set the training data before calling train_dataloader.")
        dataset = TensorDataset(self.train_data[0], self.train_data[1])
        return DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=True)

    def val_dataloader(self):
        if self.val_data is None:
            raise ValueError("Validation data not set. Please set the validation data before calling val_dataloader.")
        dataset = TensorDataset(self.val_data[0], self.val_data[1])
        return DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=False)

# Example usage
if __name__ == "__main__":
    config = get_deep_learning_config()  # Replace it with the desired configuration function
    model = DeepLearningModule(config)

    # Set training and validation data
    train_data = (torch.randn(1000, 1, 28, 28), torch.randint(0, 10, (1000,)))
    val_data = (torch.randn(200, 1, 28, 28), torch.randint(0, 10, (200,)))
    model.train_data = train_data
    model.val_data = val_data

    trainer = pl.Trainer(max_epochs=config['training']['epochs'])
    trainer.fit(model)
