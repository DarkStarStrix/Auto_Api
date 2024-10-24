from setuptools import setup, find_packages

setup(
    name="lightning-automl",
    version="0.1.0",
    description="Automated Machine Learning Pipeline with PyTorch Lightning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "feature-engine>=1.0.0",  # For automated feature engineering
        "optuna>=2.10.0",  # For hyperparameter optimization
        "tensorboard>=2.12.0",  # For logging
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
        ],
    },
    python_requires=">=3.8",
)
