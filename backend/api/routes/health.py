"""Health and readiness checks."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from backend.config import ARTIFACT_DIR, OUTPUT_DIR

router = APIRouter(prefix="/api/v1", tags=["System"])


@router.get("/health")
async def health_check():
    artifacts_ok = all(
        (ARTIFACT_DIR / name).exists()
        for name in (
            "feature_scaler.pkl",
            "graphsage.pt",
            "isolation_forest.pkl",
            "autoencoder.pt",
            "risk_model.pkl",
        )
    )

    default_output = (OUTPUT_DIR / "graphs" / "graph_edges.csv").exists()

    return {
        "status": "healthy",
        "artifacts_ready": artifacts_ok,
        "default_output_available": default_output,
        "pipeline_stages": [
            "Data Ingestion",
            "Identity Resolution",
            "Graph Construction",
            "Feature Engineering",
            "Rule Engine",
            "GNN Engine",
            "Isolation Forest",
            "Autoencoder",
            "Feature Fusion",
            "Risk Model Inference",
            "Fusion & Scoring",
            "Explainability",
            "Alert Generation",
        ],
    }
