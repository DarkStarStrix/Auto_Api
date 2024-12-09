import torch.utils.data as data
import numpy as np
from typing import Dict, Any
import torch
import pytorch_lightning as pl
from Model_Types import NaiveBayesModel
from Model_Library import get_naive_bayes_config
import os
import gc


def create_naive_bayes_model(config: Dict [str, Any]):
    """Create NaiveBayes model from configuration."""
    if config ['model'] ['type'] != 'naive_bayes':
        raise ValueError (f"Expected naive_bayes model type, got: {config ['model'] ['type']}")
    return NaiveBayesModel (config)


def train_naive_bayes(X: np.ndarray, y: np.ndarray, config: Dict [str, Any]):
    """Train a Naive Bayes model with given data and configuration."""
    try:
        if torch.cuda.is_available ():
            torch.cuda.empty_cache ()
        gc.collect ()

        X_tensor = torch.FloatTensor (X)
        y_tensor = torch.FloatTensor (y)
        dataset = data.TensorDataset (X_tensor, y_tensor)

        train_size = int ((1 - config ['training'] ['validation_split']) * len (dataset))
        val_size = len (dataset) - train_size
        generator = torch.Generator ().manual_seed (42)
        train_dataset, val_dataset = data.random_split (
            dataset,
            [train_size, val_size],
            generator=generator
        )

        dataloader_kwargs = {
            'batch_size': config ['training'] ['batch_size'],
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
        }

        train_loader = data.DataLoader (
            train_dataset,
            shuffle=True,
            **dataloader_kwargs
        )

        val_loader = data.DataLoader (
            val_dataset,
            shuffle=False,
            **dataloader_kwargs
        )

        model = create_naive_bayes_model (config)

        trainer = pl.Trainer (
            max_epochs=config ['training'] ['epochs'],
            accelerator="auto",
            devices=1,
            callbacks=model.configure_callbacks (),
            enable_progress_bar=True,
            logger=config ['logging'] ['tensorboard'],
            log_every_n_steps=config ['logging'] ['log_interval'],
            enable_checkpointing=True,
            deterministic=True,
            detect_anomaly=True,
            precision=32,
        )

        trainer.fit (model, train_loader, val_loader)
        return model

    except Exception as e:
        print (f"Error in train_naive_bayes: {str (e)}")
        raise


def main():
    """Main function to demonstrate Naive Bayes training."""
    try:
        torch.manual_seed (42)
        np.random.seed (42)
        if torch.cuda.is_available ():
            torch.cuda.manual_seed (42)

        n_samples = 500
        n_features = 10
        n_classes = 3

        X = np.random.randn (n_samples, n_features)
        y = np.random.randint (0, n_classes, n_samples)

        config = get_naive_bayes_config ()
        config ['model'].update ({
            'input_dim': n_features,
            'output_dim': n_classes
        })

        print ("Starting Naive Bayes training...")
        model = train_naive_bayes (X, y, config)
        print ("Training completed successfully!")

        save_path = 'models'
        os.makedirs (save_path, exist_ok=True)
        torch.save (
            model.state_dict (),
            os.path.join (save_path, 'naive_bayes_model.pt')
        )
        print ("Model saved successfully!")

    except Exception as e:
        print (f"Error in main: {str (e)}")
        raise
    finally:
        if torch.cuda.is_available ():
            torch.cuda.empty_cache ()
        gc.collect ()


if __name__ == "__main__":
    main ()
