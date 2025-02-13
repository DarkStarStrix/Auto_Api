import torch.utils.data as data
import numpy as np
from typing import Dict, Any
import torch
import pytorch_lightning as pl
import os
import gc
from torch.utils.data import TensorDataset, DataLoader
from Model_Types import KMeansModel
from Model_Library import get_kmeans_config


def create_kmeans_model(config: Dict [str, Any]):
    """Create KMeans model from configuration."""
    if config ['model'] ['type'] != 'kmeans':
        raise ValueError (f"Expected kmeans model type, got: {config ['model'] ['type']}")
    return KMeansModel (config)


def train_kmeans(X: np.ndarray, config: Dict [str, Any]):
    """Train a KMeans model with given data and configuration."""
    try:
        if torch.cuda.is_available ():
            torch.cuda.empty_cache ()
        gc.collect ()

        X = np.asarray (X, dtype=np.float32)

        X_tensor = torch.from_numpy (X)

        num_samples = len (X_tensor)
        indices = torch.randperm (num_samples)

        train_size = int ((1 - config ['training'] ['validation_split']) * num_samples)
        train_indices = indices [:train_size]
        val_indices = indices [train_size:]

        X_train = X_tensor [train_indices]
        X_val = X_tensor [val_indices]

        model = create_kmeans_model (config)

        train_dataset = TensorDataset (X_train, X_train)
        val_dataset = TensorDataset (X_val, X_val)

        train_loader = DataLoader (
            train_dataset,
            batch_size=config ['training'].get ('batch_size', 32),
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        val_loader = DataLoader (
            val_dataset,
            batch_size=config ['training'].get ('batch_size', 32),
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

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
        print (f"Error in train_kmeans: {str (e)}")
        raise


def main():
    """Main function to demonstrate KMeans training."""
    try:
        torch.manual_seed (42)
        np.random.seed (42)
        if torch.cuda.is_available ():
            torch.cuda.manual_seed (42)

        n_samples = 500
        n_features = 10
        n_clusters = 3

        centers = np.random.randn (n_clusters, n_features) * 5
        X = np.vstack ([
            np.random.randn (n_samples // n_clusters, n_features) + center
            for center in centers
        ])

        np.random.shuffle (X)

        print ("Starting KMeans training...")
        kmeans_config = get_kmeans_config ()
        kmeans_config ['model'].update ({
            'input_dim': n_features,
            'n_clusters': n_clusters
        })
        model = train_kmeans (X, kmeans_config)
        print ("KMeans training completed successfully!")

        save_path = 'models'
        os.makedirs (save_path, exist_ok=True)
        torch.save (
            model.state_dict (),
            os.path.join (save_path, 'kmeans_model.pt')
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
