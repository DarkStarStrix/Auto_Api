import torch.utils.data as data
import numpy as np
from Model_Library import get_linear_regression_config
from typing import Dict, Any
import torch
import pytorch_lightning as pl
from Model_Types import LinearRegressionModel, LogisticRegressionModel
import os
import gc


def create_model(config: Dict [str, Any]):
    """Create the appropriate model based on configuration."""
    model_type = config ['model'] ['type']
    if model_type == 'logistic_regression':
        return LogisticRegressionModel (config)
    elif model_type == 'linear_regression':
        return LinearRegressionModel (config)
    else:
        raise ValueError (f"Unknown model type: {model_type}")


def train_model(X: np.ndarray, y: np.ndarray, config: Dict [str, Any]):
    """Train a model with the given data and configuration."""
    try:
        if torch.cuda.is_available ():
            torch.cuda.empty_cache ()
        gc.collect ()

        X_tensor = torch.FloatTensor (X)
        y_tensor = torch.FloatTensor (y.reshape (-1, 1))

        dataset = data.TensorDataset (X_tensor, y_tensor)

        train_size = int (0.8 * len (dataset))
        val_size = len (dataset) - train_size

        generator = torch.Generator ().manual_seed (42)

        train_dataset, val_dataset = data.random_split (
            dataset,
            [train_size, val_size],
            generator=generator
        )

        dataloader_kwargs = {
            'batch_size': config ['training'].get ('batch_size', 32),
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

        model = create_model (config)

        trainer = pl.Trainer (
            max_epochs=config ['training'].get ('epochs', 10),
            accelerator="auto",
            devices=1,
            callbacks=model.configure_callbacks (),
            enable_progress_bar=True,
            logger=True,
            log_every_n_steps=10,
            enable_checkpointing=True,
            deterministic=True,
            max_steps=config ['training'].get ('max_steps', -1),
            detect_anomaly=True,
            gradient_clip_val=config ['training'].get ('gradient_clip_val', 0.5),
            accumulate_grad_batches=config ['training'].get ('accumulate_grad_batches', 1),
            precision=32,
        )

        trainer.fit (model, train_loader, val_loader)

        return model

    except Exception as e:
        print (f"Error in train_model: {str (e)}")
        raise


def main():
    """Main function to run the training."""
    try:
        torch.manual_seed (42)
        np.random.seed (42)
        if torch.cuda.is_available ():
            torch.cuda.manual_seed (42)

        X = np.random.randn (500, 10)  # Features
        true_coefficients = np.random.randn (10)
        y = np.dot (X, true_coefficients) + np.random.normal (0, 0.1, size=500)

        config = get_linear_regression_config ()
        config ['model'] ['input_dim'] = X.shape [1]
        config ['model'] ['output_dim'] = 1
        config ['model'] ['type'] = 'linear_regression'

        config ['training'].update ({
            'batch_size': 16,
            'gradient_clip_val': 0.5,
            'accumulate_grad_batches': 2,
            'max_steps': 100
        })

        print ("Starting training...")
        model = train_model (X, y, config)
        print ("Training completed successfully!")

        save_path = 'models'
        os.makedirs (save_path, exist_ok=True)
        torch.save (
            model.state_dict (),
            os.path.join (save_path, 'linear_model.pt')  # Updated model name
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
