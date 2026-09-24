"""LLM-powered natural language wallet explanation routes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.api.schemas import LLMExplanationResponse
from backend.api.services.data_loader import DataLoader
from backend.config import OUTPUT_DIR
from backend.database import get_job
from backend.llm_agent.explanation_generator import ExplanationGenerator

router = APIRouter(prefix="/api/v1", tags=["LLM Explanation"])


def _get_loader_for_job(
    job_id: Optional[str],
) -> tuple[DataLoader, str]:
    """
    Resolve a DataLoader and effective job_id.

    Lightweight version that does NOT eagerly build the graph or
    pattern detector — only the data loader is needed for explanations.
    """
    if not job_id:
        default_root = OUTPUT_DIR
        if not (default_root / "features" / "wallet_features.csv").exists():
            raise HTTPException(
                status_code=404,
                detail="No default pipeline output found. Upload data via POST /api/v1/jobs/upload first.",
            )
        return DataLoader(default_root), "__default__"

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] == "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' failed: {job.get('error_message')}",
        )
    if job["status"] == "processing":
        raise HTTPException(
            status_code=202,
            detail=(
                f"Job '{job_id}' is currently "
                f"{job.get('current_stage', 'processing')} "
                f"({job.get('progress', 0)}%)."
            ),
        )

    loader = DataLoader(Path(job["output_dir"]), job_id=job_id)
    return loader, job_id


@router.get(
    "/jobs/{job_id}/explain/wallet/{wallet_id}",
    response_model=LLMExplanationResponse,
)
async def get_job_llm_explanation(job_id: str, wallet_id: str):
    """Generate or retrieve a cached natural language explanation for a wallet."""
    loader, effective_job_id = _get_loader_for_job(job_id)
    generator = ExplanationGenerator(loader, job_id=effective_job_id)
    try:
        result = generator.explain(wallet_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return LLMExplanationResponse(**result)


@router.get(
    "/explain/wallet/{wallet_id}",
    response_model=LLMExplanationResponse,
)
async def get_global_llm_explanation(
    wallet_id: str,
    job_id: Optional[str] = Query(None),
):
    """Generate or retrieve a cached natural language explanation for a wallet."""
    loader, effective_job_id = _get_loader_for_job(job_id)
    generator = ExplanationGenerator(loader, job_id=effective_job_id)
    try:
        result = generator.explain(wallet_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return LLMExplanationResponse(**result)
