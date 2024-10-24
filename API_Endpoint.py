# main.py
from fastapi import FastAPI, HTTPException, Request
import requests
import torch
from lightning_auto import AutoML

app = FastAPI ()


# Define the API endpoint for training
@app.post ("/api/train")
async def api_train(request: Request):
    try:
        # Get the configuration from the request
        config = await request.json ()

        # Create an instance of AutoML with the provided configuration
        auto_ml = AutoML (config)

        # Generate example training and validation data
        train_features = torch.randn (1000, config ["model"] ["input_dim"])
        train_labels = torch.randint (0, config ["model"] ["output_dim"], (1000,))
        val_features = torch.randn (200, config ["model"] ["input_dim"])
        val_labels = torch.randint (0, config ["model"] ["output_dim"], (200,))

        train_data = torch.utils.data.DataLoader (
            torch.utils.data.TensorDataset (train_features, train_labels),
            batch_size=config ["data"] ["batch_size"],
            shuffle=True
        )
        val_data = torch.utils.data.DataLoader (
            torch.utils.data.TensorDataset (val_features, val_labels),
            batch_size=config ["data"] ["batch_size"]
        )

        # Train the model
        auto_ml.fit (train_data, val_data)

        # Save the model
        model_path = "model.pt"
        torch.save (auto_ml.model.state_dict (), model_path)

        return {"message": "Training completed successfully", "model_path": model_path}
    except Exception as e:
        raise HTTPException (status_code=500, detail=str (e))

