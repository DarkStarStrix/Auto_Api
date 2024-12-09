from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import jwt
import uuid
from Model_Library import (
    get_linear_config,
    get_logistic_regression_config,
    get_transformer_config,
    get_cnn_config
)
from lightning_auto import AutoML

app = FastAPI ()
security = HTTPBearer ()

# Store active jobs
jobs = {}


class TrainingRequest (BaseModel):
    config_name: str
    parameters: Dict [str, Any]


class JobStatus (BaseModel):
    job_id: str
    status: str
    start_time: datetime
    end_time: Optional [datetime]
    metrics: Optional [Dict [str, float]]


def get_config_templates():
    return {
        "linear": get_linear_config (),
        "logistic": get_logistic_regression_config (),
        "transformer": get_transformer_config (),
        "cnn": get_cnn_config ()
    }


async def train_model(job_id: str, config: Dict [str, Any]):
    try:
        jobs [job_id].status = "training"
        model = AutoML (config)
        model.fit (config ["data"])
        jobs [job_id].status = "completed"
        jobs [job_id].end_time = datetime
        jobs [job_id].metrics = model.get_metrics ()
    except Exception as e:
        jobs [job_id].status = "failed"
        jobs [job_id].end_time = datetime
        jobs [job_id].metrics = {"error": str (e)}


@app.get ("/api/v1/config/templates")
async def get_configs():
    return {"templates": get_config_templates ()}


@app.post ("/api/v1/train")
async def start_training(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        token: HTTPAuthorizationCredentials = Depends (security)
):
    templates = get_config_templates ()
    if request.config_name not in templates:
        raise HTTPException (status_code=400, detail="Invalid config name")

    job_id = str (uuid.uuid4 ())
    jobs [job_id] = JobStatus (
        job_id=job_id,
        status="initializing",
        start_time=datetime,
        metrics=None,
        end_time=datetime,
    )

    config = templates [request.config_name]
    config.update (request.parameters)

    background_tasks.add_task (train_model, job_id, config)
    return {"job_id": job_id}


@app.get ("/api/v1/jobs/{job_id}")
async def get_job_status(
        job_id: str,
        token: HTTPAuthorizationCredentials = Depends (security)
):
    if job_id not in jobs:
        raise HTTPException (status_code=404, detail="Job not found")
    return jobs [job_id]


@app.delete ("/api/v1/jobs/{job_id}")
async def stop_job(
        job_id: str,
        token: HTTPAuthorizationCredentials = Depends (security)
):
    if job_id not in jobs:
        raise HTTPException (status_code=404, detail="Job not found")
    if jobs [job_id].status == "training":
        # Implement emergency stop logic
        jobs [job_id].status = "stopped"
        jobs [job_id].end_time = datetime
    return {"status": "stopped"}


@app.post ("/api/v1/token")
async def generate_token(api_key: str):
    # Implement your authentication logic
    token = jwt.encode (
        {"user_id": str (uuid.uuid4 ())},
        "your-secret-key",
        algorithm="HS256"
    )
    return {"token": token}
