"""Alert retrieval routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.api.dependencies import get_services_for_job
from backend.api.schemas import AlertsResponse
from backend.database import get_job

router = APIRouter(prefix="/api/v1", tags=["Alerts"])


def _load_alerts(loader, job_id: str) -> AlertsResponse:
    alerts_data = loader.alerts

    return AlertsResponse(
        job_id=job_id,
        status="completed",
        total_alerts=len(alerts_data),
        alerts=alerts_data,
    )


@router.get("/jobs/{job_id}/alerts", response_model=AlertsResponse)
async def get_job_alerts(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] == "processing":
        raise HTTPException(
            status_code=202,
            detail=f"Job '{job_id}' still processing ({job.get('progress', 0)}%).",
        )
    if job["status"] == "failed":
        raise HTTPException(status_code=400, detail=job.get("error_message", "Job failed."))

    loader, _, _, _, _ = get_services_for_job(job_id)
    return _load_alerts(loader, job_id)


@router.get("/alerts", response_model=AlertsResponse)
async def get_global_alerts(job_id: Optional[str] = Query(None)):
    loader, _, _, _, resolved_id = get_services_for_job(job_id)
    return _load_alerts(loader, resolved_id or "default")
