"""Job upload and status routes."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from backend.api.schemas import JobStatusResponse, JobSummaryResponse, JobUploadResponse
from backend.api.worker import run_job_background
from backend.config import JOBS_DIR, OPTIONAL_RAW_FILES, REQUIRED_RAW_FILES
from backend.database import create_job, get_job, list_jobs

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])

EXPECTED_FILES = {f: Path(f).stem for f in REQUIRED_RAW_FILES}
OPTIONAL_FILE_MAP = {f: Path(f).stem for f in OPTIONAL_RAW_FILES}


@router.post("/upload", response_model=JobUploadResponse)
async def upload_raw_csv_job(request: Request, background_tasks: BackgroundTasks):
    """Upload raw CSV files and trigger the end-to-end forensic pipeline."""
    uploaded: dict[str, object] = {}

    try:
        form = await request.form()
        for key, val in form.multi_items():
            if not hasattr(val, "filename") or not val.filename:
                continue
            clean_fname = val.filename.lower().strip()
            clean_key = key.lower().strip()
            for exp_name in {**EXPECTED_FILES, **OPTIONAL_FILE_MAP}:
                stem = exp_name.replace(".csv", "")
                if (
                    clean_fname == exp_name
                    or clean_fname.endswith(f"_{exp_name}")
                    or clean_key == stem
                    or clean_key == exp_name
                ):
                    uploaded.setdefault(exp_name, val)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse upload: {exc}") from exc

    missing = [f for f in REQUIRED_RAW_FILES if f not in uploaded]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required CSV files: {missing}. Required: {REQUIRED_RAW_FILES}",
        )

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    raw_dir = job_dir / "raw"
    output_dir = job_dir / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for filename, up_file in uploaded.items():
            content = await up_file.read()  # type: ignore[union-attr]
            (raw_dir / filename).write_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save uploads: {exc}") from exc

    create_job(
        job_id=job_id,
        raw_dir=str(raw_dir),
        output_dir=str(output_dir),
        status="processing",
        current_stage="Data Ingestion",
        progress=5,
    )

    background_tasks.add_task(run_job_background, job_id, raw_dir, output_dir)

    return JobUploadResponse(
        job_id=job_id,
        status="processing",
        message="Pipeline started: ingestion → identity → graph → features → detectors → fusion → SHAP → alerts.",
    )


@router.get("", response_model=List[JobStatusResponse])
async def list_all_jobs(limit: int = 50, offset: int = 0):
    jobs = list_jobs(limit=limit, offset=offset)
    return [_to_status(j) for j in jobs]


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_by_id(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _to_status(job)


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    return await get_job_by_id(job_id)


@router.get("/{job_id}/summary", response_model=JobSummaryResponse)
async def get_job_summary(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != "completed":
        return JobSummaryResponse(job_id=job_id, status=job["status"])

    summary = job.get("summary") or {}
    dataset = summary.get("dataset", {})
    alerts_sum = summary.get("alerts_summary", {})
    high_c = alerts_sum.get("high_risk_count", 0)
    med_c = alerts_sum.get("medium_risk_count", 0)
    low_c = alerts_sum.get("low_risk_count", 0)

    return JobSummaryResponse(
        job_id=job_id,
        status="completed",
        total_wallets=dataset.get("wallet_count", alerts_sum.get("total_alerts", 0)),
        anomalies_found=high_c + med_c,
        high_risk_count=high_c,
        medium_risk_count=med_c,
        low_risk_count=low_c,
        execution_time_seconds=summary.get("execution_time_seconds"),
        details=summary,
    )


@router.get("/{job_id}/manifest")
async def get_job_manifest(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    manifest_path = Path(job["output_dir"]) / "reports" / "pipeline_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found for this job.")
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _to_status(job: dict) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0) or (100 if job["status"] == "completed" else 0),
        step=job.get("current_stage") or job["status"],
        current_stage=job.get("current_stage"),
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at"),
        error_message=job.get("error_message"),
        summary=job.get("summary"),
    )
