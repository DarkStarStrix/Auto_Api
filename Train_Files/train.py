"""
Example workflow showing how to use AutoML.
Users can copy and modify this file for their use case.
"""

from lightning_auto import AutoML
from Model_Library import get_linear_config
import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    # Get configuration
    config = get_linear_config()

    # Optionally modify config for your specific case
    config ["model"] ["output_dim"] = 5  # Example modification

    # Create automated ML pipeline
    auto_ml = AutoML (config)

    # Example data
    train_features = torch.randn (1000, config ["model"] ["input_dim"])
    train_labels = torch.randint (0, config ["model"] ["output_dim"], (1000,))
    val_features = torch.randn (200, config ["model"] ["input_dim"])
    val_labels = torch.randint (0, config ["model"] ["output_dim"], (200,))

    # Create DataLoader
    train_data = DataLoader (TensorDataset (train_features, train_labels), batch_size=config ["data"] ["batch_size"],
                             shuffle=True)
    val_data = DataLoader (TensorDataset (val_features, val_labels), batch_size=config ["data"] ["batch_size"])

    # Start training
    auto_ml.fit (train_data, val_data)

    # Save model if needed
    torch.save (auto_ml.model.state_dict (), "model.pt")


if __name__ == "__main__":
    main ()
