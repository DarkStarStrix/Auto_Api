import os
import gc
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from Model_Library import get_linear_regression_config
from Model_Types import LinearRegressionModel


def create_model(config: Dict [str, Any]):
    """Create the appropriate model based on configuration."""
    model_type = config ['model'] ['type']
    if model_type == 'linear_regression':
        return LinearRegressionModel (config)
    else:
        raise ValueError (f"Unknown model type: {model_type}")


def train_model(X: np.ndarray, y: np.ndarray, config: Dict[str, Any]):
    """Train a model with the given data and configuration."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print(X[:5])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)

        train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = data.TensorDataset(X_val_tensor, y_val_tensor)

        dataloader_kwargs = {
            'batch_size': config['training'].get('batch_size', 32),
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
        }

        train_loader = data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
        val_loader = data.DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

        model = create_model(config)

        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="auto",
            devices=1,
            callbacks=model.configure_callbacks(),
            enable_progress_bar=True,
            logger=True,
            log_every_n_steps=10,
            enable_checkpointing=True,
            deterministic=True,
            detect_anomaly=True,
            gradient_clip_val=config['training'].get('gradient_clip_val', 0.5),
            accumulate_grad_batches=1,
            precision=32,
        )

        trainer.fit(model, train_loader, val_loader)

        return model

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        raise


def main():
    """Main function to run the training."""
    try:
        torch.manual_seed (42)
        np.random.seed (42)
        if torch.cuda.is_available ():
            torch.cuda.manual_seed (42)

        data = pd.read_csv ('C:/Users/kunya/PycharmProjects/Auto_Api/Data/processed_advertising.csv')
        X = data [['TV']].values
        y = data ['Sales'].values

        config = get_linear_regression_config ()
        config ['model'] ['input_dim'] = X.shape [1]
        config ['model'] ['type'] = 'linear_regression'

        config ['training'].update ({
            'batch_size': 16,
            'gradient_clip_val': 0.5,
            'accumulate_grad_batches': 2,
            'max_steps': 100,
            'learning_rate': 1
        })

        print ("Starting training...")
        model = train_model (X, y, config)
        print ("Training completed successfully!")

        save_path = 'models'
        os.makedirs (save_path, exist_ok=True)
        torch.save (model.state_dict (), os.path.join (save_path, 'model.pt'))
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
