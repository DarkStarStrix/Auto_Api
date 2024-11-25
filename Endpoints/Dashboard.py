from fastapi import APIRouter, Depends
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel
from Auth import get_current_user
import json

router = APIRouter ()


class JobSummary (BaseModel):
    job_id: str
    status: str
    start_time: datetime
    model_type: str
    metrics: Dict = {}


jobs = {}


@router.get ("/dashboard/jobs")
async def get_user_jobs(user=Depends (get_current_user)) -> List [JobSummary]:
    user_jobs = []
    for job in jobs.values ():
        if job.get ('user_id') == user ['username']:
            user_jobs.append (JobSummary (
                job_id=job ['job_id'],
                status=job ['status'],
                start_time=job ['start_time'],
                model_type=job ['model_type'],
                metrics=job.get ('metrics', {})
            ))
    return user_jobs


@router.get ("/dashboard/metrics")
async def get_user_metrics(user=Depends (get_current_user)):
    user_jobs = [j for j in jobs.values () if j.get ('user_id') == user ['username']]
    return {
        "total_jobs": len (user_jobs),
        "completed_jobs": len ([j for j in user_jobs if j ['status'] == "completed"]),
        "active_jobs": len ([j for j in user_jobs if j ['status'] == "training"])
    }


@router.get ("/dashboard/templates")
async def get_available_templates():
    with open ('static/templates.json', 'r') as f:
        return json.load (f)
