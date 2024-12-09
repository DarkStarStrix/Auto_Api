import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Model_Types import LinearRegressionModel
from Model_Library import get_linear_regression_config
import os


def load_and_preprocess_data(data_path: str):
    # Load the CSV data
    data = pd.read_csv(data_path)

    # Select features and target
    features = ['beds']
    target = 'price'

    # Split into features and target
    X = data[features].values
    y = data[target].values

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def train_model(X: np.ndarray, y: np.ndarray, config: dict):
    # Convert data to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # Create dataset and data loaders
    dataset = TensorDataset(X_tensor, y_tensor)
    train_data, val_data = train_test_split(dataset, train_size=0.8, random_state=42)
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'])

    # Create model
    model = LinearRegressionModel(config)

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        devices=1,
        accelerator="auto",
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    return model


def main():
    # Set random seeds for reproducibility
    pl.seed_everything(42)

    # Load and preprocess data
    data_path = 'C:/Users/kunya/PycharmProjects/Auto_Api/data/Training Data set 2 - train.csv'
    X, y = load_and_preprocess_data(data_path)

    # Get and update config
    config = get_linear_regression_config()
    config['model']['input_dim'] = X.shape[1]

    # Train the model
    model = train_model(X, y, config)

    # Save the trained model
    save_path = 'models'
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'housing_price_model.pt'))
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
