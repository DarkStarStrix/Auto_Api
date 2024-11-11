# utils/config_templates.py
def get_config_templates():
    """Predefined configuration templates for different model types"""
    return {
        "linear_regression": {
            "model_type": "linear_regression",
            "parameters": {
                "learning_rate": 0.001,
                "optimizer": "adam",
                "loss": "mse",
                "metrics": ["mae", "mse"],
                "batch_size": 32,
                "epochs": 100,
                "validation_split": 0.2
            }
        },
        "logistic_regression": {
            "model_type": "logistic_regression",
            "parameters": {
                "learning_rate": 0.001,
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "metrics": ["accuracy", "precision", "recall"],
                "batch_size": 32,
                "epochs": 100,
                "validation_split": 0.2
            }
        }
    }
