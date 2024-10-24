"""
Configuration templates with model-specific optimizations.
"""


def get_linear_config():
    """Get optimized configuration with fixed scheduler setup."""
    return {
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
        },
        "training": {
            "learning_rate": 0.002,
            "epochs": 30,
            "batch_size": 32,
            "early_stopping": True,
            "patience": 7
        },
        "optimization": {
            "optimizer": "adam",
            "scheduler": "cosine",  # Changed back to cosine for stability
            "weight_decay": 0.01,
            "min_lr": 1e-6,
            "gradient_clip_val": 1.0,
            "mixed_precision": True
        },
        "data": {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True
        },
        "logging": {
            "tensorboard": True,
            "log_every_n_steps": 10,
            "metrics": ["accuracy"]
        }
    }


def get_transformer_config():
    """Configuration for transformer models."""
    return {
        "model": {
            "type": "transformer",
            "input_dim": 768,
            "hidden_dim": 512,
            "output_dim": 10,
            "num_layers": 6
        },
        "training": {
            "learning_rate": 1e-4,
            "epochs": 20,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "early_stopping": True
        },
        "optimization": {
            "mixed_precision": True,
            "gradient_checkpointing": True,
            "accumulate_grad_batches": 4,
            "compile": True
        },
        "data": {
            "batch_size": 32,
            "num_workers": 8
        }
    }


def get_cnn_config():
    """Configuration for CNN models."""
    return {
        "model": {
            "type": "cnn",
            "input_dim": [3, 224, 224],
            "output_dim": 1000
        },
        "training": {
            "learning_rate": 1e-3,
            "epochs": 30,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "early_stopping": True
        },
        "optimization": {
            "mixed_precision": True,
            "gradient_checkpointing": False,
            "accumulate_grad_batches": 2,
            "compile": True
        },
        "data": {
            "batch_size": 128,
            "num_workers": 8
        }
    }
