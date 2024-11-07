# train_2.py

import torch.utils.data as data
import numpy as np
from Config import get_logistic_regression_config
from typing import Dict, Any
import torch
import pytorch_lightning as pl
from Model_Types import LogisticRegressionModel, LinearRegressionModel
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
        # Clear any existing tensors from GPU memory
        if torch.cuda.is_available ():
            torch.cuda.empty_cache ()
        gc.collect ()

        # Convert data to tensors (on CPU)
        X_tensor = torch.FloatTensor (X)
        y_tensor = torch.FloatTensor (y)

        # Create dataset
        dataset = data.TensorDataset (X_tensor, y_tensor)

        # Create data loaders with optimized settings
        train_size = int (0.8 * len (dataset))
        val_size = len (dataset) - train_size

        # Use a fixed seed for reproducibility
        generator = torch.Generator ().manual_seed (42)

        train_dataset, val_dataset = data.random_split (
            dataset,
            [train_size, val_size],
            generator=generator
        )

        # Optimized DataLoader settings
        dataloader_kwargs = {
            'batch_size': config ['training'].get ('batch_size', 32),
            'num_workers': 0,  # Set to 0 to avoid multiprocessing issues
            'pin_memory': False,  # Set to False to reduce memory usage
            'persistent_workers': False,  # Set to False as we're not using workers
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

        # Create model
        model = create_model (config)

        # Configure trainer with memory optimizations
        trainer = pl.Trainer (
            max_epochs=config ['training'].get ('epochs', 10),
            accelerator="auto",  # Let PyTorch Lightning choose the best accelerator
            devices=1,
            callbacks=model.configure_callbacks (),
            enable_progress_bar=True,
            logger=True,
            log_every_n_steps=10,
            enable_checkpointing=True,
            deterministic=True,  # For reproducibility
            max_steps=config ['training'].get ('max_steps', -1),
            detect_anomaly=True,  # For debugging
            # Memory optimizations
            gradient_clip_val=config ['training'].get ('gradient_clip_val', 0.5),
            accumulate_grad_batches=config ['training'].get ('accumulate_grad_batches', 1),
            precision=32,  # Use 32-bit precision for stability
        )

        # Train model
        trainer.fit (model, train_loader, val_loader)

        return model

    except Exception as e:
        print (f"Error in train_model: {str (e)}")
        raise


def main():
    """Main function to run the training."""
    try:
        # Set random seeds for reproducibility
        torch.manual_seed (42)
        np.random.seed (42)
        if torch.cuda.is_available ():
            torch.cuda.manual_seed (42)

        # Generate smaller sample data for testing
        X = np.random.randn (500, 10)  # Reduced sample size
        y = (X.sum (axis=1) > 0).astype (float)

        # Get and update config
        config = get_logistic_regression_config ()
        config ['model'] ['input_dim'] = X.shape [1]

        # Add memory optimization settings
        config ['training'].update ({
            'batch_size': 16,  # Smaller batch size
            'gradient_clip_val': 0.5,
            'accumulate_grad_batches': 2,
            'max_steps': 100  # Limit training steps for testing
        })

        print ("Starting training...")
        model = train_model (X, y, config)
        print ("Training completed successfully!")

        # Save the model
        save_path = 'models'
        os.makedirs (save_path, exist_ok=True)
        torch.save (
            model.state_dict (),
            os.path.join (save_path, 'model.pt')
        )
        print ("Model saved successfully!")

    except Exception as e:
        print (f"Error in main: {str (e)}")
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available ():
            torch.cuda.empty_cache ()
        gc.collect ()


if __name__ == "__main__":
    main ()
