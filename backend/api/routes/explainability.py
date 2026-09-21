"""SHAP explainability routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from backend.api.dependencies import get_services_for_job
from backend.api.schemas import ExplainabilityResponse

router = APIRouter(prefix="/api/v1", tags=["Explainability"])


@router.get(
    "/jobs/{job_id}/explainability/wallet/{wallet_id}",
    response_model=ExplainabilityResponse,
)
async def get_job_explainability(job_id: str, wallet_id: str):
    _, _, explain_engine, _, _ = get_services_for_job(job_id)
    return ExplainabilityResponse(**explain_engine.explain_wallet_risk(wallet_id))


@router.get("/explainability/wallet/{wallet_id}", response_model=ExplainabilityResponse)
async def get_global_explainability(
    wallet_id: str,
    job_id: Optional[str] = Query(None),
):
    _, _, explain_engine, _, _ = get_services_for_job(job_id)
    return ExplainabilityResponse(**explain_engine.explain_wallet_risk(wallet_id))
