import torch.utils.data as data
import numpy as np
from typing import Dict, Any
import torch
import pytorch_lightning as pl
from Model_Types import SVMModel
from Model_Library import get_svm_config
import os
import gc


def create_svm_model(config: Dict[str, Any]):
    """Create an SVM model from configuration."""
    if config['model']['type'] != 'svm':
        raise ValueError(f"Expected svm model type, got: {config['model']['type']}")
    return SVMModel(config)


def train_svm(X: np.ndarray, y: np.ndarray, config: Dict[str, Any]):
    """Train an SVM model with given data and configuration."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = data.TensorDataset(X_tensor, y_tensor)

        train_size = int((1 - config['training']['validation_split']) * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = data.random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )

        dataloader_kwargs = {
            'batch_size': config['training']['batch_size'],
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
        }

        train_loader = data.DataLoader(
            train_dataset,
            shuffle=True,
            **dataloader_kwargs
        )

        val_loader = data.DataLoader(
            val_dataset,
            shuffle=False,
            **dataloader_kwargs
        )

        model = create_svm_model(config)

        trainer = pl.Trainer(
            max_epochs=config['training']['epochs'],
            accelerator="auto",
            devices=1,
            callbacks=model.configure_callbacks(),
            enable_progress_bar=True,
            logger=config['logging']['tensorboard'],
            log_every_n_steps=config['logging']['log_interval'],
            enable_checkpointing=True,
            deterministic=True,
            detect_anomaly=True,
            precision=32,
        )

        trainer.fit(model, train_loader, val_loader)
        return model

    except Exception as e:
        print(f"Error in train_svm: {str(e)}")
        raise


def main():
    """Main function to demonstrate SVM training."""
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        n_samples = 500
        n_features = 10
        n_classes = 2

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)

        print("Starting SVM training...")
        svm_config = get_svm_config()
        svm_config['model'].update({
            'input_dim': n_features,
            'output_dim': n_classes
        })
        model = train_svm(X, y, svm_config)
        print("SVM training completed successfully!")

        save_path = 'models'
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(save_path, 'svm_model.pt')
        )
        print("Model saved successfully!")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
