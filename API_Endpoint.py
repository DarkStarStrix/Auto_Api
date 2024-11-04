import requests
from lightning_auto import AutoML


def main():
    # Authenticate with the API
    api_token = get_api_token()

    # Get the configuration template
    config = get_config_template("linear")

    # Optionally modify the configuration
    config["model"]["output_dim"] = 5

    # Start the training process
    response = requests.post(
        "https://api.automl.dev/v1/train",
        json={"config_name": "linear", "config_overrides": config["model"]},
        headers={"Authorization": f"Bearer {api_token}"}
    )
    job_id = response.json()["job_id"]

    # Monitor the training progress
    model_info = requests.get(
        f"https://api.automl.dev/v1/models/{job_id}",
        headers={"Authorization": f"Bearer {api_token}"}
    ).json()
    print(f"Training status: {model_info['status']}")
    print(f"Training metrics: {model_info['metrics']}")

    # Save the model
    torch.save(auto_ml.model.state_dict(), "model.pt")


def get_api_token():
    response = requests.post(
        "https://api.automl.dev/v1/auth/token",
        json={"username": "your_username", "password": "your_password"}
    )
    return response.json()["access_token"]


def get_config_template(config_name):
    response = requests.get(
        f"https://api.automl.dev/v1/configs/{config_name}",
    )
    return response.json()["config"]


if __name__ == "__main__":
    main()

