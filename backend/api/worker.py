"""Background pipeline worker invoked by the API after file upload."""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from backend.config import ARTIFACT_DIR
from backend.database import update_job
from backend.pipeline.orchestrator import run_pipeline

logger = logging.getLogger(__name__)

STAGE_PROGRESS = {
    "Data Ingestion": 8,
    "Identity Resolution": 12,
    "Graph Construction": 20,
    "Feature Engineering": 30,
    "Rule Engine": 42,
    "GNN Engine": 52,
    "Isolation Forest": 62,
    "Autoencoder": 70,
    "Feature Fusion": 76,
    "Risk Model Inference": 82,
    "Fusion & Scoring": 88,
    "Explainability": 93,
    "Alert Generation": 97,
    "Pipeline Completed": 100,
}


def run_job_background(job_id: str, raw_dir: Path, output_dir: Path) -> None:
    """Execute the full forensic pipeline for a job and persist status updates."""
    try:
        logger.info("Starting pipeline for job %s", job_id)

        def on_progress(stage_name: str, _fraction: float) -> None:
            update_job(
                job_id,
                current_stage=stage_name,
                progress=STAGE_PROGRESS.get(stage_name, 50),
            )

        manifest = run_pipeline(
            raw_dir=raw_dir,
            output_dir=output_dir,
            artifact_dir=ARTIFACT_DIR,
            progress_callback=on_progress,
        )

        update_job(
            job_id,
            status="completed",
            current_stage="completed",
            progress=100,
            completed_at=datetime.now(timezone.utc).isoformat(),
            summary=manifest,
        )
        logger.info("Job %s completed successfully", job_id)
    except Exception as exc:
        logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        update_job(
            job_id,
            status="failed",
            current_stage="failed",
            progress=0,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error_message=str(exc),
        )
