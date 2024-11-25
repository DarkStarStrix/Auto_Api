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
            "num_workers": 7,
            "pin_memory": True
        },
        "logging": {
            "tensorboard": True,
            "log_every_n_steps": 10,
            "metrics": ["accuracy"]
        }
    }


def get_linear_regression_config():
    """Get optimized configuration for a linear regression model."""
    return {
        "model": {
            "type": "regression",
            "input_dim": 10,
            "output_dim": 1,
            "task": "regression"
        },
        "training": {
            "learning_rate": 0.01,
            "epochs": 100,
            "batch_size": 32,
            "early_stopping": True,
            "patience": 10
        },
        "optimization": {
            "optimizer": "adam",
            "scheduler": "cosine",
            "weight_decay": 0.001,
            "min_lr": 1e-6,
            "gradient_clip_val": 1.0,
            "mixed_precision": True
        },
        "data": {
            "batch_size": 32,
            "num_workers": 7,
            "pin_memory": True
        },
        "logging": {
            "tensorboard": True,
            "log_every_n_steps": 10,
            "metrics": ["mse", "r2"]
        }
    }


def get_logistic_regression_config():
    """Get configuration for a logistic regression model with optimal hyperparameters."""
    return {
        "model": {
            "type": "logistic_regression",
            "input_dim": 10,  # Adjust based on your data
            "output_dim": 1,  # Binary classification
            "task": "binary_classification"
        },
        "training": {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "early_stopping": True,
            "patience": 5,
            "validation_split": 0.2,
            "gradient_clip_val": 1.0,
            "visualization": True,
            "metrics": ["accuracy", "precision", "recall", "f1"]
        },
        "optimization": {
            "optimizer": {
                "type": "adam",
                "betas": (0.9, 0.999),
                "epsilon": 1e-8,
                "weight_decay": 0.01,
            },
            "scheduler": {
                "type": "cosine",
                "min_lr": 1e-6,
                "warmup_epochs": 3
            },
            "mixed_precision": True,
            "gradient_clip": True,
            "gradient_clip_val": 1.0,
            "gradient_accumulation_steps": 1
        },
        "data": {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "preprocessing": {
                "normalize": True,
                "handle_missing": "mean",
                "handle_categorical": "one_hot"
            }
        },
        "logging": {
            "tensorboard": True,
            "log_every_n_steps": 10,
            "save_dir": "logs/logistic_regression",
            "metrics": ["accuracy", "f1", "precision", "recall"],
            "save_visualization": True
        },
        "hyperparameters": {
            "weight_initialization": "xavier_uniform",
            "dropout_rate": 0.0,  # Usually not needed for simple logistic regression
            "bias_init": 0.0
        }
    }


def refined_get_logistic_regression_config(
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        optimizer_type="adam",
        scheduler_type="cosine",
        enable_hyperparameter_tuning=False,
        enable_quantization=False
):
    """Get logistic regression configuration with optimization options."""
    return {
        "model": {
            "type": "logistic_regression",
            "input_dim": 10,
            "output_dim": 1,
            "task": "binary_classification"
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "early_stopping": True,
            "patience": 5,
            "validation_split": 0.2,
            "gradient_clip_val": 1.0,
            "visualization": True,
            "metrics": ["accuracy", "precision", "recall", "f1"]
        },
        "optimization": {
            "optimizer": {
                "type": optimizer_type,
                "betas": (0.9, 0.999) if optimizer_type == "adam" else None,
                "epsilon": 1e-8,
                "weight_decay": 0.01,
            },
            "scheduler": {
                "type": scheduler_type,
                "min_lr": 1e-6,
                "warmup_epochs": 3
            },
            "mixed_precision": True,
            "gradient_clip": True,
            "gradient_clip_val": 1.0,
            "gradient_accumulation_steps": 1,
            "hyperparameter_tuning": {
                "enabled": enable_hyperparameter_tuning,
                "n_trials": 20,
                "search_space": {
                    "learning_rate": (1e-5, 1e-1),
                    "batch_size": (16, 128),
                    "optimizer": ["adam", "sgd", "adamw"],
                    "weight_decay": (1e-5, 1e-2)
                },
                "optimization_metric": "val_loss"
            },
            "quantization": {
                "enabled": enable_quantization,
                "dtype": "qint8",
                "modules_to_quantize": ["Linear"],
                "calibration_method": "histogram",
                "reduce_range": True,
                "backend": "fbgemm"
            },
        },
        "data": {
            "batch_size": batch_size,
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "preprocessing": {
                "normalize": True,
                "handle_missing": "mean",
                "handle_categorical": "one_hot"
            }
        },
        "logging": {
            "tensorboard": True,
            "log_every_n_steps": 10,
            "save_dir": "logs/logistic_regression",
            "metrics": ["accuracy", "f1", "precision", "recall"],
            "save_visualization": True
        },
        "hyperparameters": {
            "weight_initialization": "xavier_uniform",
            "dropout_rate": 0.0,
            "bias_init": 0.0
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
