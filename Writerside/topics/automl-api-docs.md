# AutoML API Documentation

A powerful and flexible API for automating machine learning workflows. This service provides endpoints for model training, configuration management, and job monitoring with secure authentication.


> This API requires authentication. Ensure you have an API key before starting. Contact your administrator or visit the dashboard to generate one.

##  Quick Start

```python
from automl_client import AutoMLClient

# Initialize client
client = AutoMLClient("http://your-domain:8000", api_key="your-api-key")
client.authenticate()

# Start training
config = {
    "model_type": "linear_regression",
    "data_config": {
        "input_path": "data.csv",
        "target_column": "target"
    }
}

job = client.start_training(config)
print(f"Training started! Job ID: {job['job_id']}")
```


> Check out the `examples/` directory for more detailed usage examples and notebooks.

##  Installation

### Docker Deployment
```bash
# Clone the repository
git clone https://github.com/your-repo/automl-api
cd automl-api

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start services
docker-compose up -d
```

### Client Installation
```bash
# Install Python client
pip install automl-client

# Install CLI tool
pip install automl-cli
```

##  Authentication

### Getting Started with Authentication
1. Generate an API key from the dashboard
2. Use the key to get an authentication token
3. Include the token in all later requests

```python
# Using Python client
client = AutoMLClient("http://your-domain:8000", api_key="your-api-key")
token = client.authenticate()

# Using CLI
automl login --save
```


> Keep your API key secure and never commit it to version control.

## API Endpoints

### Configuration Templates
GET `/api/v1/config/templates`
```python
# Get available templates
templates = client.get_templates()
```

### Training
POST `/api/v1/train`
```python
# Start training job
config = {
    "model_type": "linear_regression",
    "data_config": {
        "input_path": "data.csv",
        "target_column": "target"
    }
}
job = client.start_training(config)
```

### Job Monitoring
GET `/api/v1/jobs/{job_id}`
```python
# Check job status
status = client.get_job_status(job_id)
```

## Command Line Interface

The CLI provides easy access to common operations:

```bash
# List available templates
automl templates

# Start training
automl train --template linear_regression --data-path data.csv --target y

# Monitor job
automl status <job_id>
```

## Web Dashboard

Access the web interface at `http://your-domain:8000/dashboard`

Features:
- Model template browsing
- Job monitoring
- Performance metrics visualization
- Configuration management

##  Configuration

### Model Templates
Available templates include:

1. Linear Regression
```python
{
    "model_type": "linear_regression",
    "parameters": {
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "mse",
        "metrics": ["mae", "mse"],
        "batch_size": 32,
        "epochs": 100
    }
}
```

2. Logistic Regression
```python
{
    "model_type": "logistic_regression",
    "parameters": {
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy"],
        "batch_size": 32,
        "epochs": 100
    }
}
```

> Parameters can be customized based on your specific needs.

##  Monitoring and Metrics

### Available Metrics
- Training loss
- Validation metrics
- Learning rate
- Job status
- Resource usage

### Monitoring Methods
1. Web Dashboard
2. CLI Commands
3. Python Client

```python
# Using Python client
metrics = client.get_job_metrics(job_id)
```

```bash
# Using CLI
automl metrics <job_id>
```

## Troubleshooting

> [!WARNING]
> Common issues and solutions:

1. Authentication Failed
```bash
# Check API key validity
automl verify-key

# Regenerate token
automl login --refresh
```

2. Job Failed
```python
# Get detailed error information
status = client.get_job_status(job_id)
print(status.get('error'))
```

##  Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request


> Please follow our coding standards and include tests with your contributions.

## Support

- GitHub Issues: [Report a bug](https://github.com/DarkStarStrix/Auto_Api/issues)
- Documentation: [Full documentation](https://github.com/DarkStarStrix/Auto_Api/tree/master/Writerside)
