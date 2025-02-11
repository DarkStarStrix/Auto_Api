from setuptools import setup, find_packages

with open ("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read ()

setup (
    name="auto-ml-api",
    version="0.1.0",
    description="Automated Machine Learning Pipeline with API Interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages (),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core ML
        "torch==2.2.0",
        "pytorch-lightning==2.1.1",
        "transformers==4.48.0",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "scipy==1.13.1",

        # API and Web
        "fastapi==0.109.1",
        "uvicorn==0.15.0",
        "requests==2.31.0",
        "pydantic==1.10.18",
        "python-jose==3.3.0",
        "python-dotenv==1.0.0",
        "PyJWT==2.10.1",

        # ML Tools
        "scikit-learn==1.3.0",
        "seaborn==0.11.2",
        "matplotlib==3.7.1",
        "optuna==4.1.0",
        "config==0.5.1",
        "plotly==5.24.1",

        # Monitoring and Logging
        "prometheus-client==0.21.0",
        "rich==13.9.0",
        "click==8.1.7",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "ml": [
            "lightgbm>=4.0.0",
            "xgboost>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "automl=automl.cli:main",
        ],
    },
)
