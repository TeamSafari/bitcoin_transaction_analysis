"""Graph visualization routes for Cytoscape.js."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from backend.api.dependencies import get_services_for_job
from backend.api.schemas import GraphOverviewResponse, GraphResponse

router = APIRouter(prefix="/api/v1", tags=["Graph"])


@router.get("/jobs/{job_id}/graph/overview", response_model=GraphOverviewResponse)
async def get_job_graph_overview(job_id: str, max_nodes: int = Query(200, ge=10, le=2000)):
    _, _, _, explorer, _ = get_services_for_job(job_id)
    return GraphOverviewResponse(**explorer.get_overview_graph(max_nodes=max_nodes))


@router.get("/graph/overview", response_model=GraphOverviewResponse)
async def get_global_graph_overview(
    job_id: Optional[str] = Query(None),
    max_nodes: int = Query(200, ge=10, le=2000),
):
    _, _, _, explorer, _ = get_services_for_job(job_id)
    return GraphOverviewResponse(**explorer.get_overview_graph(max_nodes=max_nodes))


@router.get("/jobs/{job_id}/graph/wallet/{wallet_id}", response_model=GraphResponse)
async def get_job_wallet_ego_graph(
    job_id: str,
    wallet_id: str,
    hops: int = Query(2, ge=1, le=5),
):
    _, _, _, explorer, _ = get_services_for_job(job_id)
    return GraphResponse(**explorer.get_ego_graph(wallet_id, hops=hops))


@router.get("/graph/wallet/{wallet_id}", response_model=GraphResponse)
async def get_global_wallet_ego_graph(
    wallet_id: str,
    job_id: Optional[str] = Query(None),
    hops: int = Query(2, ge=1, le=5),
):
    _, _, _, explorer, _ = get_services_for_job(job_id)
    return GraphResponse(**explorer.get_ego_graph(wallet_id, hops=hops))
