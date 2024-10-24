# AutoML Pipeline API Documentation

## Base URL
```
https://api.automl.dev/v1
```

## Authentication
```bash
curl -H "Authorization: Bearer <YOUR_API_KEY>" \
     -X POST https://api.automl.dev/v1/train
```

## Endpoints

### Training
```
POST /train
```
Start training with configuration:
```json
{
    "config": {
        "model": {
            "type": "classification",
            "input_dim": 10
        }
    },
    "data": {
        "train_path": "s3://bucket/train.csv",
        "val_path": "s3://bucket/val.csv"
    }
}
```

### Configurations
```
GET /configs
```
List available configurations:
```json
{
    "configs": [
        "classification",
        "regression",
        "vision"
    ]
}
```

### Models
```
GET /models/{model_id}
```
Get model information:
```json
{
    "model_id": "m123",
    "status": "training",
    "metrics": {
        "train_loss": 1.6422,
        "val_loss": 1.6169
    }
}
```

## Using the API
Example Python client:
```python
import requests

class AutoMLClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.automl.dev/v1"

    def train(self, config):
        response = requests.post(
            f"{self.base_url}/train",
            json={"config": config},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()
```
