import torch.utils.data as data
import numpy as np
from typing import Dict, Any
import torch
import pytorch_lightning as pl
from Model_Types import DecisionTreeModel
from Model_Library import get_decision_tree_config
import os
import gc


def create_decision_tree_model(config: Dict [str, Any]):
    """Create DecisionTree model from configuration."""
    if config ['model'] ['type'] != 'decision_tree':
        raise ValueError (f"Expected decision_tree model type, got: {config ['model'] ['type']}")
    return DecisionTreeModel (config)


def train_decision_tree(X: np.ndarray, y: np.ndarray, config: Dict [str, Any]):
    """Train a Decision Tree model with given data and configuration."""
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

        model = create_decision_tree_model (config)

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
        print (f"Error in train_decision_tree: {str (e)}")
        raise


def main():
    """Main function to demonstrate Decision Tree training."""
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

        config = get_decision_tree_config ()
        config ['model'].update ({
            'input_dim': n_features,
            'output_dim': n_classes,
            'max_depth': 5
        })

        print ("Starting Decision Tree training...")
        model = train_decision_tree (X, y, config)
        print ("Training completed successfully!")

        save_path = 'models'
        os.makedirs (save_path, exist_ok=True)
        torch.save (
            model.state_dict (),
            os.path.join (save_path, 'decision_tree_model.pt')
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
