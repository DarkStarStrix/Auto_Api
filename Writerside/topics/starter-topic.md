# About How it works

The provided code is part of an automated machine learning pipeline implemented using PyTorch Lightning. The `AutoML` class is the core component, designed to handle various aspects of the machine learning workflow, including data preprocessing, model creation, training, and visualization.

### Initialization and Model Creation

The `AutoML` class is initialized with a configuration dictionary and an optional model. If no model is provided, the `_create_model` method constructs a default model based on the configuration. The model architecture varies depending on whether the task is classification or regression:

```python
def _create_model(self):
    input_dim = self.config['model']['input_dim']
    hidden_dim = self.config['model']['hidden_dim']
    output_dim = self.config['model']['output_dim']

    if self.config['model']['task'] == 'classification':
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    else:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
```

### Data Pipeline Setup

The `_setup_data_pipeline` method initializes the data pipeline when training starts. It configures the data loader settings such as batch size, number of workers, and prefetch factor. The method also processes and prepares the training and validation datasets:

```python
def _setup_data_pipeline(self, train_data, val_data):
    data_config = self.config.get('data', {})
    self.data_pipeline = {
        'batch_size': data_config.get('batch_size', 32),
        'num_workers': data_config.get('num_workers', 7),
        'pin_memory': data_config.get('pin_memory', True),
        'shuffle': data_config.get('shuffle', True),
        'drop_last': data_config.get('drop_last', False),
        'prefetch_factor': data_config.get('prefetch_factor', 2),
    }
    self.train_dataset = self._prepare_dataset(train_data, is_train=True)
    self.val_dataset = self._prepare_dataset(val_data, is_train=False) if val_data is not None else None
```

### Training and Validation Steps

The `training_step` and `validation_step` methods define the operations performed during each training and validation step, respectively. Both methods use automatic mixed precision (AMP) for efficient computation and log relevant metrics:

```python
def training_step(self, batch, batch_idx):
    with autocast(enabled=self.use_amp):
        loss = self._compute_loss(batch)
    self.log('train_loss', loss, prog_bar=True)
    self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)
    return loss

def validation_step(self, batch, batch_idx):
    with autocast(enabled=self.use_amp):
        loss = self._compute_loss(batch)
    self.log('val_loss', loss, prog_bar=True)
    return loss
```

### Visualization Callback

The `VisualizationCallback` class is a custom callback for creating and saving visualizations during training. It collects metrics such as training and validation losses, learning rates, and gradients, and generates plots for these metrics. The `_plot_model_predictions` method visualizes model predictions against actual values:

```python
def _plot_model_predictions(self, pl_module):
    try:
        num_samples = 100
        input_dim = pl_module.config['model']['input_dim']
        output_dim = pl_module.config['model']['output_dim']
        example_data = torch.randn(num_samples, input_dim)
        with torch.no_grad():
            logits = pl_module.model(example_data)
            predictions = logits.argmax(dim=1).cpu().numpy()
        actual_values = np.array([i % output_dim for i in range(num_samples)])
        plt.figure(figsize=(10, 8))
        plt.scatter(actual_values, predictions, alpha=0.5)
        plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'r--')
        plt.title('Model Predictions vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predictions')
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'model_predictions.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot model predictions. Error: {str(e)}")
```

### Optimizer and Scheduler Configuration

The `configure_optimizers` method sets up the optimizer and learning rate scheduler. It uses the Adam optimizer and a cosine annealing learning rate scheduler, which adjusts the learning rate over epochs:

```python
def configure_optimizers(self):
    opt_config = self.config.get('optimization', {})
    optimizer = torch.optim.Adam(
        self.parameters(),
        lr=self.config['training'].get('learning_rate', 0.001),
        weight_decay=opt_config.get('weight_decay', 0.01)
    )
    scheduler_config = {
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training'].get('epochs', 30),
            eta_min=opt_config.get('min_lr', 1e-6)
        ),
        "interval": "epoch",
        "frequency": 1,
        "monitor": "val_loss"
    }
    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler_config
    }
```

Overall, the `AutoML` class and its associated methods and callbacks provide a comprehensive framework for automating the machine learning workflow, from data preprocessing to model training and visualization.

## config.py

The provided code in `Config.py` defines configuration templates for different types of machine learning models, including linear models, transformers, and convolutional neural networks (CNNs). Each configuration is tailored to optimize the performance of the respective model type.

### Linear Model Configuration

The `get_linear_config` function returns a configuration dictionary optimized for linear models, particularly for classification tasks. The model configuration specifies the input, hidden, and output dimensions, along with architectural details such as dropout rate, activation function, number of layers, and batch normalization:

```python
"model": {
    "type": "classification",
    "input_dim": 10,
    "hidden_dim": 256,
    "output_dim": 5,
    "task": "classification",
    "architecture": {
        "dropout_rate": 0.3,
        "activation": "relu",
        "num_layers": 3,
        "batch_norm": True
    }
}
```

The training configuration includes parameters like learning rate, number of epochs, batch size, and early stopping criteria:

```python
"training": {
    "learning_rate": 0.002,
    "epochs": 30,
    "batch_size": 32,
    "early_stopping": True,
    "patience": 7
}
```

The optimization settings specify the optimizer type, learning rate scheduler, weight decay, and gradient clipping value:

```python
"optimization": {
    "optimizer": "adam",
    "scheduler": "cosine",
    "weight_decay": 0.01,
    "min_lr": 1e-6,
    "gradient_clip_val": 1.0,
    "mixed_precision": True
}
```

### Transformer Model Configuration

The `get_transformer_config` function provides a configuration for transformer models. This configuration includes the input dimension, hidden dimension, output dimension, and the number of layers:

```python
"model": {
    "type": "transformer",
    "input_dim": 768,
    "hidden_dim": 512,
    "output_dim": 10,
    "num_layers": 6
}
```

The training settings for transformers include a lower learning rate and the use of the AdamW optimizer:

```python
"training": {
    "learning_rate": 1e-4,
    "epochs": 20,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "early_stopping": True
}
```

The optimization settings for transformers include mixed precision, gradient checkpointing, and gradient accumulation:

```python
"optimization": {
    "mixed_precision": True,
    "gradient_checkpointing": True,
    "accumulate_grad_batches": 4,
    "compile": True
}
```

### CNN Model Configuration

The `get_cnn_config` function returns a configuration for CNN models. The model configuration specifies the input dimensions (e.g., image dimensions) and the output dimension:

```python
"model": {
    "type": "cnn",
    "input_dim": [3, 224, 224],
    "output_dim": 1000
}
```

The training settings for CNN include a higher learning rate and the use of the AdamW optimizer:

```python
"training": {
    "learning_rate": 1e-3,
    "epochs": 30,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "early_stopping": True
}
```

The optimization settings for CNN include mixed precision and gradient accumulation:

```python
"optimization": {
    "mixed_precision": True,
    "gradient_checkpointing": False,
    "accumulate_grad_batches": 2,
    "compile": True
}
```

Overall, these configuration templates provide a structured way to define and optimize the settings for different types of machine learning models, ensuring that each model type is configured with appropriate parameters for optimal performance.

## Train.py
The `train.py` file demonstrates an example workflow for using the `AutoML` class to automate the machine learning pipeline. The script is designed to be easily modifiable for different use cases.

### Configuration Setup

The script begins by importing necessary modules and functions, including `AutoML` from `lightning_auto` and `get_linear_config` from `Config`. The `main` function is defined to encapsulate the workflow.

```python
from lightning_auto import AutoML
from Config import get_linear_config
import torch
from torch.utils.data import DataLoader, TensorDataset
```

### Main Function

Within the `main` function, the configuration for a linear model is retrieved using `get_linear_config`. The configuration can be modified as needed, such as changing the output dimension:

```python
config = get_linear_config()
config["model"]["output_dim"] = 5  # Example modification
```

### AutoML Pipeline Creation

An instance of the `AutoML` class is created using the modified configuration. This instance will handle the entire machine learning pipeline, from data preprocessing to model training and evaluation:

```python
auto_ml = AutoML(config)
```

### Data Preparation

Example training and validation data are generated using PyTorch's `randn` and `randint` functions. These datasets are then wrapped in `DataLoader` objects to facilitate batch processing during training:

```python
train_features = torch.randn(1000, config["model"]["input_dim"])
train_labels = torch.randint(0, config["model"]["output_dim"], (1000,))
val_features = torch.randn(200, config["model"]["input_dim"])
val_labels = torch.randint(0, config["model"]["output_dim"], (200,))

train_data = DataLoader(TensorDataset(train_features, train_labels), batch_size=config["data"]["batch_size"], shuffle=True)
val_data = DataLoader(TensorDataset(val_features, val_labels), batch_size=config["data"]["batch_size"])
```

### Model Training

The `fit` method of the `AutoML` instance is called to start the training process. This method takes the training and validation data loaders as input and handles the training loop, validation, and logging:

```python
auto_ml.fit(train_data, val_data)
```

### Model Saving

After training, the model's state dictionary is saved to a file named `model.pt` using PyTorch's `save` function. This allows the trained model to be loaded and used later:

```python
torch.save(auto_ml.model.state_dict(), "model.pt")
```

### Entry Point

Finally, the script includes a standard Python entry point check to ensure that the `main` function is called when the script is executed directly:

```python
if __name__ == "__main__":
    main()
```

This structure makes the script modular and easy to adapt for different machine learning tasks by modifying the configuration and data preparation steps.
