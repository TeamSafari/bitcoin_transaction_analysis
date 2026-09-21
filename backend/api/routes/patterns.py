"""Fund-flow trace pattern routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from backend.api.dependencies import get_services_for_job
from backend.api.schemas import TraceResponse, TransactionStep

router = APIRouter(prefix="/api/v1", tags=["Patterns"])


def _build_trace(wallet_id: str, pattern_engine, max_hops: int) -> TraceResponse:
    pattern, confidence, sequence = pattern_engine.detect_pattern(wallet_id, max_hops=max_hops)
    steps = [
        TransactionStep(
            step=tx["step"],
            source_wallet=tx["source_wallet"],
            source=tx["source_wallet"],
            target_wallet=tx["target_wallet"],
            target=tx["target_wallet"],
            txid=tx["txid"],
            amount_sats=tx["amount_sats"],
            timestamp=tx.get("timestamp", ""),
        )
        for tx in sequence
    ]
    return TraceResponse(
        wallet_id=wallet_id,
        pattern=pattern,
        pattern_detected=pattern,
        confidence_score=confidence,
        sequence=steps,
        total_amount_sats=sum(tx["amount_sats"] for tx in sequence),
        hop_count=len(steps),
    )


@router.get("/jobs/{job_id}/patterns/trace/{wallet_id}", response_model=TraceResponse)
async def get_job_patterns_trace(
    job_id: str,
    wallet_id: str,
    max_hops: int = Query(5, ge=1, le=10),
):
    _, pattern_engine, _, _, _ = get_services_for_job(job_id)
    return _build_trace(wallet_id, pattern_engine, max_hops)


@router.get("/patterns/trace/{wallet_id}", response_model=TraceResponse)
async def get_global_patterns_trace(
    wallet_id: str,
    job_id: Optional[str] = Query(None),
    max_hops: int = Query(5, ge=1, le=10),
):
    _, pattern_engine, _, _, _ = get_services_for_job(job_id)
    return _build_trace(wallet_id, pattern_engine, max_hops)
