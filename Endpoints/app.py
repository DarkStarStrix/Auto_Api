# app.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import jwt
import uuid
from datetime import datetime, timedelta

from models.base_model import BaseModel
from train import get_linear_config
from config_templates import get_config_templates
from metrics_callback import MetricsCallback

app = FastAPI (title="AutoML API", version="0.0.2")
security = HTTPBearer ()


# Configuration Models
class ConfigTemplate (BaseModel):
    model_type: str
    parameters: Dict [str, Any]


class TrainingConfig (BaseModel):
    model_type: str
    config: Dict [str, Any]
    data_config: Dict [str, Any]


class JobStatus (BaseModel):
    job_id: str
    status: str
    metrics: Optional [Dict [str, float]]
    created_at: datetime
    updated_at: datetime


# Token Management
def create_api_token(user_id: str) -> str:
    expiration = datetime.utcnow () + timedelta (days=30)
    return jwt.encode (
        {"user_id": user_id, "exp": expiration},
        "your-secret-key",  # Move to environment variable
        algorithm="HS256"
    )


def verify_token(credentials: HTTPAuthorizationCredentials = Depends (security)):
    try:
        payload = jwt.decode (
            credentials.credentials, "your-secret-key", algorithms=["HS256"]
        )
        return payload ["user_id"]
    except:
        raise HTTPException (status_code=401, detail="Invalid token")


# Global job storage (replace with a database in production)
active_jobs = {}


# API Routes
@app.post ("/api/v1/token")
async def generate_token(api_key: str):
    """Generate API token for authentication"""
    # Validate API key against database
    if api_key == "valid-api-key":  # Replace it with actual validation
        token = create_api_token (str (uuid.uuid4 ()))
        return {"token": token}
    raise HTTPException (status_code=401, detail="Invalid API key")


@app.get ("/api/v1/config/templates")
async def get_templates(_: str = Depends (verify_token)):
    """Get available configuration templates"""
    templates = get_config_templates ()
    return {"templates": templates}


@app.post ("/api/v1/train")
async def start_training(
        config: TrainingConfig,
        background_tasks: BackgroundTasks,
        user_id: str = Depends (verify_token)
):
    """Start a new training job"""
    job_id = str (uuid.uuid4 ())

    # Initialize job tracking
    active_jobs [job_id] = JobStatus (
        job_id=job_id,
        status="initializing",
        metrics={},
        created_at=datetime.utcnow (),
        updated_at=datetime.utcnow ()
    )

    # Start training in background
    background_tasks.add_task (
        run_training_job,
        job_id=job_id,
        config=config
    )

    return {"job_id": job_id, "status": "initializing"}


@app.get ("/api/v1/jobs/{job_id}")
async def get_job_status(
        job_id: str,
        _: str = Depends (verify_token)
):
    """Get the status of a specific job"""
    if job_id not in active_jobs:
        raise HTTPException (status_code=404, detail="Job not found")
    return active_jobs [job_id]


@app.get ("/api/v1/jobs")
async def list_jobs(
        user_id: str = Depends (verify_token),
        limit: int = 10,
        offset: int = 0
):
    """List all jobs for the user"""
    # In production, filter by user_id from a database
    jobs = list (active_jobs.values ())
    return {
        "jobs": jobs [offset:offset + limit],
        "total": len (jobs)
    }


# Training job handler
async def run_training_job(job_id: str, config: TrainingConfig):
    """Background task for model training"""
    try:
        # Update job status
        active_jobs [job_id].status = "training"
        active_jobs [job_id].updated_at = datetime()

        # Initialize training components
        metrics_callback = MetricsCallback ()
        training_loop = TrainingLoop (config.config)

        # Run training
        model = await training_loop.train (
            config.data_config,
            metrics_callback=metrics_callback
        )

        # Update a job with final metrics
        active_jobs [job_id].status = "completed"
        active_jobs [job_id].metrics = metrics_callback.get_final_metrics ()

    except Exception as e:
        active_jobs [job_id].status = "failed"
        active_jobs [job_id].metrics = {"error": str (e)}

    finally:
        active_jobs [job_id].updated_at = datetime()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run (app, host="0.0.0.0", port=8000)
