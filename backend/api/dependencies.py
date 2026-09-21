"""FastAPI dependency helpers for resolving job-scoped services."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException

from backend.api.services.data_loader import DataLoader
from backend.api.services.explainability_engine import ExplainabilityEngine
from backend.api.services.gnn_influence import GNNInfluenceCalculator
from backend.api.services.graph_explorer import GraphExplorer
from backend.api.services.pattern_detector import PatternDetector
from backend.config import OUTPUT_DIR
from backend.database import get_job


def get_services_for_job(
    job_id: Optional[str],
) -> Tuple[DataLoader, PatternDetector, ExplainabilityEngine, GraphExplorer, Optional[str]]:
    """
    Resolve data services for a job output directory.

    When job_id is None, uses the default demo output directory if it exists.
    """
    if not job_id:
        default_root = OUTPUT_DIR
        if not (default_root / "graphs" / "graph_edges.csv").exists():
            raise HTTPException(
                status_code=404,
                detail="No default pipeline output found. Upload data via POST /api/v1/jobs/upload first.",
            )
        loader = DataLoader(default_root)
        influence = GNNInfluenceCalculator(loader)
        return (
            loader,
            PatternDetector(loader),
            ExplainabilityEngine(loader),
            GraphExplorer(loader, influence),
            None,
        )

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

    loader = DataLoader(Path(job["output_dir"]))
    influence = GNNInfluenceCalculator(loader)
    pattern = PatternDetector(loader, raw_dir=job.get("raw_dir"))
    explain = ExplainabilityEngine(loader)
    graph = GraphExplorer(loader, influence)
    return loader, pattern, explain, graph, job_id
