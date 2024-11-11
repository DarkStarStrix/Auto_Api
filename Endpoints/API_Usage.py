import requests


class AutoMLClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip ('/')
        self.api_key = api_key
        self.token = None

    def authenticate(self):
        """Get authentication token"""
        response = requests.post (
            f"{self.base_url}/api/v1/token",
            json={"api_key": self.api_key}
        )
        self.token = response.json () ["token"]

    def get_templates(self):
        """Get available configuration templates"""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get (
            f"{self.base_url}/api/v1/config/templates",
            headers=headers
        )
        return response.json () ["templates"]

    def start_training(self, config):
        """Start a new training job"""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post (
            f"{self.base_url}/api/v1/train",
            headers=headers,
            json=config
        )
        return response.json ()

    def get_job_status(self, job_id: str):
        """Get status of a specific job"""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get (
            f"{self.base_url}/api/v1/jobs/{job_id}",
            headers=headers
        )
        return response.json ()


# Usage example
if __name__ == "__main__":
    client = AutoMLClient ("http://localhost:8000", "your-api-key")
    client.authenticate ()

    # Get templates
    templates = client.get_templates ()

    # Start training with linear regression template
    config = {
        "model_type": "linear_regression",
        "config": templates ["linear_regression"] ["parameters"],
        "data_config": {
            "input_path": "data/training.csv",
            "target_column": "target",
            "features": ["feature1", "feature2"]
        }
    }

    job = client.start_training (config)
    job_id = job ["job_id"]

    # Monitor job status
    while True:
        status = client.get_job_status (job_id)
        print (f"Job status: {status ['status']}")
        if status ['status'] in ['completed', 'failed']:
            break
        time.sleep (10)
