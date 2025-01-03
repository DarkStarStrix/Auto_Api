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


def get_CNN_RNN_config():
    """Configuration template for training basic CNN-RNN hybrid model."""
    return {
        "model": {
            # CNN Parameters
            "conv_layers": [32, 64, 128],  # Number of filters in each conv layer
            "kernel_size": 3,
            "pool_size": 2,
            "cnn_activation": "relu",

            # RNN Parameters
            "rnn_type": "LSTM",  # LSTM or GRU
            "hidden_size": 128,
            "num_rnn_layers": 2,
            "bidirectional": True,

            # General Architecture
            "input_channels": 1,
            "dropout_rate": 0.5,
            "final_dense_layers": [256, 128],
            "output_size": 10  # Number of classes
        },

        "training": {
            "num_epochs": 100,
            "batch_size": 32,
            "shuffle": True,
            "validation_split": 0.2,
            "early_stopping_patience": 10,
            "save_best_only": True,
            "monitor_metric": "val_accuracy"
        },

        "optimization": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "lr_scheduler": "reduce_on_plateau",
            "lr_patience": 5,
            "lr_factor": 0.1,
            "min_lr": 1e-6,
            "clip_grad_norm": 1.0
        },

        "data": {
            "dataset": "MNIST",
            "data_dir": "./data",
            "num_workers": 4,
            "pin_memory": True,
            "normalize": True,
            "augmentation": {
                "random_rotation": 10,
                "random_zoom": 0.1,
                "width_shift": 0.1,
                "height_shift": 0.1
            }
        }
    }


def get_decision_tree_config():
    return {
        "model": {
            "type": "decision_tree",
            "input_dim": None,  # Set dynamically
            "output_dim": None,  # Set dynamically
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "auto",
            "criterion": "gini",
            "task": "classification"
        },
        "training": {
            "batch_size": 32,
            "epochs": 10,
            "validation_split": 0.2,
            "early_stopping": True,
            "patience": 3
        },
        "preprocessing": {
            "scaling": None,  # Trees don't require scaling
            "handle_missing": "median",
            "handle_categorical": "label_encoding"
        },
        "logging": {
            "tensorboard": True,
            "log_interval": 10,
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "feature_importance": True
        }
    }


def get_naive_bayes_config():
    return {
        "model": {
            "type": "naive_bayes",
            "input_dim": None,  # Set dynamically based on data
            "output_dim": None,  # Set dynamically based on data
            "var_smoothing": 1e-9,  # Smoothing parameter
            "priors": None,  # Class priors, None for automatic
            "task": "classification"
        },
        "training": {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "early_stopping": True,
            "patience": 5,
            "validation_split": 0.2
        },
        "preprocessing": {
            "scaling": "standard",
            "handle_missing": "mean",
            "feature_selection": None
        },
        "logging": {
            "tensorboard": True,
            "log_interval": 10,
            "metrics": ["accuracy", "precision", "recall", "f1"]
        }
    }


def get_Thermodynamic_Diffusion_config():
    """Configuration template for training thermodynamic diffusion models."""
    return {
        "model": {
            # Model Architecture
            "image_size": 28,
            "channels": 1,
            "time_embedding_dim": 256,
            "model_channels": 64,  # Base channel multiplier
            "channel_multipliers": [1, 2, 4, 8],  # For different U-Net levels
            "num_res_blocks": 2,  # Number of residual blocks per level
            "attention_levels": [2, 3],  # Which levels to apply attention
            "dropout_rate": 0.1,
            "num_heads": 4,  # Number of attention heads

            # Diffusion Process
            "num_timesteps": 1000,
            "beta_schedule": "linear",  # Options: linear, cosine, quadratic
            "beta_start": 0.0001,
            "beta_end": 0.02,

            # Ising Model Parameters
            "temperature_schedule": "linear",  # How temperature changes during diffusion
            "initial_temperature": 0.1,
            "final_temperature": 2.0,
            "coupling_constant": 1.0,  # J in Ising model
        },

        "training": {
            # Training Process
            "num_epochs": 500,
            "save_interval": 5000,
            "eval_interval": 1000,
            "log_interval": 100,
            "sample_interval": 1000,
            "num_samples": 64,  # Number of samples to generate during evaluation

            # Loss Configuration
            "loss_type": "l2",  # Options: l1, l2, huber
            "loss_weight_type": "simple",  # Options: simple, snr, truncated

            # Sampling Configuration
            "sampling_steps": 250,  # Steps for fast sampling (< num_timesteps)
            "clip_samples": True,
            "clip_range": [-1, 1],
        },

        "optimization": {
            # Optimizer Configuration
            "optimizer": "AdamW",  # Options: Adam, AdamW, RMSprop
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "eps": 1e-8,
            "betas": (0.9, 0.999),

            # Learning Rate Scheduling
            "lr_schedule": "cosine",  # Options: cosine, linear, constant
            "warmup_steps": 5000,
            "min_lr": 1e-6,

            # Gradient Configuration
            "grad_clip": 1.0,
            "ema_decay": 0.9999,  # Exponential moving average of model weights
            "update_ema_interval": 1,
        },

        "data": {
            # Data Configuration
            "dataset": "MNIST",  # Dataset name
            "data_dir": "./data",
            "train_batch_size": 128,
            "eval_batch_size": 256,

            # Data Processing
            "num_workers": 4,
            "pin_memory": True,
            "persistence": True,

            # Augmentation
            "random_flip": False,
            "random_rotation": False,
            "normalize": True,
            "rescale": [-1, 1],  # Range to rescale images to

            # Caching
            "cache_size": 5000,  # Number of batches to cache in memory
            "prefetch_factor": 2,
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


def get_kmeans_config():
    return {
        "model": {
            "type": "kmeans",
            "input_dim": None,  # Set dynamically based on data
            "n_clusters": 3,  # Number of clusters
            "init_method": "kmeans++",  # Initialization method: 'random' or 'kmeans++'
            "task": "clustering",
            "distance_metric": "euclidean",  # Options: euclidean, manhattan, cosine
            "seed": 42  # Random seed for reproducibility
        },
        "training": {
            "batch_size": 64,
            "epochs": 50,
            "learning_rate": 0.01,
            "early_stopping": True,
            "patience": 10,
            "tolerance": 1e-4,  # Convergence criterion
            "max_no_improvement": 5,  # Maximum iterations without improvement
            "validation_split": 0.2
        },
        "preprocessing": {
            "scaling": "standard",  # Options: standard, minmax, robust, none
            "handle_missing": "mean",  # Options: mean, median, most_frequent, constant
            "dimensionality_reduction": {
                "method": None,  # Options: pca, tsne, umap, None
                "n_components": None  # Set if using dimensionality reduction
            },
            "feature_selection": None
        },
        "evaluation": {
            "metrics": [
                "inertia",
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score"
            ],
            "store_centroids": True,
            "track_cluster_evolution": True
        },
        "logging": {
            "tensorboard": True,
            "log_interval": 5,
            "save_visualizations": True,
            "visualization_types": [
                "cluster_boundaries",
                "silhouette_analysis",
                "elbow_curve",
                "cluster_sizes"
            ],
            "export_results": {
                "save_centroids": True,
                "save_assignments": True,
                "save_metrics": True
            }
        },
        "hyperparameter_search": {
            "enabled": False,
            "method": "grid",  # Options: grid, random, bayesian
            "param_grid": {
                "n_clusters": [2, 3, 4, 5, 6],
                "init_method": ["kmeans++", "random"],
                "learning_rate": [0.1, 0.01, 0.001]
            },
            "n_trials": 10,  # Number of trials for random/bayesian search
            "metric": "silhouette_score"  # Metric to optimize
        }
    }


def get_lightgbm_config():
    return {
        "model": {
            "type": "lightgbm",
            "input_dim": None,  # Set dynamically based on data
            "output_dim": None,  # Set dynamically based on data
            "num_leaves": 31,
            "max_depth": -1,  # -1 means no limit
            "min_data_in_leaf": 20,
            "boosting_type": "gbdt",
            "objective": "multiclass",  # or "binary" for binary classification
            "metric": ["multi_logloss", "multi_error"],
            "task": "classification"
        },
        "training": {
            "batch_size": 256,
            "epochs": 100,
            "learning_rate": 0.01,
            "early_stopping": True,
            "early_stopping_rounds": 10,
            "validation_split": 0.2,
            "num_boost_round": 100,
            "verbose_eval": 10
        },
        "preprocessing": {
            "scaling": None,  # LightGBM handles this internally
            "handle_missing": "default",  # LightGBM handles missing values
            "categorical_features": [],
            "feature_selection": None
        },
        "logging": {
            "tensorboard": True,
            "log_interval": 10,
            "metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "confusion_matrix"
            ]
        },
        "hyperparameter_search": {
            "enabled": False,
            "method": "optuna",
            "n_trials": 100,
            "param_grid": {
                "num_leaves": [15, 31, 63],
                "max_depth": [-1, 5, 10, 15],
                "min_data_in_leaf": [10, 20, 30, 50],
                "learning_rate": [0.01, 0.05, 0.1]
            }
        }
    }


def get_random_forest_config():
    return {
        "model": {
            "type": "random_forest",
            "input_dim": None,  # Set dynamically based on data
            "output_dim": None,  # Set dynamically based on data
            "n_estimators": 100,
            "max_depth": None,
            "task": "classification"
        },
        "training": {
            "batch_size": 64,
            "epochs": 50,
            "learning_rate": 0.01,
            "early_stopping": True,
            "patience": 10,
            "tolerance": 1e-4,
            "max_no_improvement": 5,
            "validation_split": 0.2
        },
        "preprocessing": {
            "scaling": "standard",
            "handle_missing": "mean",
            "dimensionality_reduction": {
                "method": None,
                "n_components": None
            },
            "feature_selection": None
        },
        "evaluation": {
            "metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1"
            ],
            "store_feature_importance": True
        },
        "logging": {
            "tensorboard": True,
            "log_interval": 5,
            "save_visualizations": True,
            "visualization_types": [
                "feature_importance"
            ],
            "export_results": {
                "save_metrics": True
            }
        },
        "hyperparameter_search": {
            "enabled": False,
            "method": "grid",
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30]
            },
            "n_trials": 10,
            "metric": "accuracy"
        }
    }


def get_svm_config():
    return {
        "model": {
            "type": "svm",
            "input_dim": None,  # Set dynamically based on data
            "output_dim": None,  # Set dynamically based on data
            "kernel": "rbf",
            "C": 1.0,
            "task": "classification"
        },
        "training": {
            "batch_size": 64,
            "epochs": 50,
            "learning_rate": 0.01,
            "early_stopping": True,
            "patience": 10,
            "tolerance": 1e-4,
            "max_no_improvement": 5,
            "validation_split": 0.2
        },
        "preprocessing": {
            "scaling": "standard",
            "handle_missing": "mean",
            "dimensionality_reduction": {
                "method": None,
                "n_components": None
            },
            "feature_selection": None
        },
        "evaluation": {
            "metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1"
            ]
        },
        "logging": {
            "tensorboard": True,
            "log_interval": 5,
            "save_visualizations": True,
            "visualization_types": [
                "confusion_matrix"
            ],
            "export_results": {
                "save_metrics": True
            }
        },
        "hyperparameter_search": {
            "enabled": False,
            "method": "grid",
            "param_grid": {
                "kernel": ["linear", "rbf", "poly"],
                "C": [0.1, 1.0, 10.0]
            },
            "n_trials": 10,
            "metric": "accuracy"
        }
    }


def get_gmm_config():
    return {
        "model": {
            "type": "gmm",
            "input_dim": None,  # Set dynamically based on data
            "n_components": 3,
            "covariance_type": "full",
            "task": "clustering"
        },
        "training": {
            "batch_size": 64,
            "epochs": 50,
            "learning_rate": 0.01,
            "early_stopping": True,
            "patience": 10,
            "tolerance": 1e-4,
            "max_no_improvement": 5,
            "validation_split": 0.2
        },
        "preprocessing": {
            "scaling": "standard",
            "handle_missing": "mean",
            "dimensionality_reduction": {
                "method": None,
                "n_components": None
            },
            "feature_selection": None
        },
        "evaluation": {
            "metrics": [
                "aic",
                "bic"
            ],
            "store_centroids": True
        },
        "logging": {
            "tensorboard": True,
            "log_interval": 5,
            "save_visualizations": True,
            "visualization_types": [
                "cluster_boundaries"
            ],
            "export_results": {
                "save_centroids": True,
                "save_assignments": True,
                "save_metrics": True
            }
        },
        "hyperparameter_search": {
            "enabled": False,
            "method": "grid",
            "param_grid": {
                "n_components": [2, 3, 4, 5],
                "covariance_type": ["full", "tied", "diag", "spherical"]
            },
            "n_trials": 10,
            "metric": "bic"
        }
    }
