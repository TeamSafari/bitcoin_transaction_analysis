"""
FastAPI entry point for the Bitcoin forensics backend.

Run from the backend folder:
    python main.py

Or from the repo root:
    python -m backend.main
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Allow `python main.py` when cwd is backend/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import alerts, explainability, graph, health, jobs, llm_explain, patterns
from backend.config import ARTIFACT_DIR
from backend.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_ARTIFACTS = (
    "feature_scaler.pkl",
    "graphsage.pt",
    "isolation_forest.pkl",
    "autoencoder.pt",
    "risk_model.pkl",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    missing = [name for name in REQUIRED_ARTIFACTS if not (ARTIFACT_DIR / name).exists()]
    if missing:
        logger.warning("Missing model artifacts: %s", missing)
    else:
        logger.info("All model artifacts present in %s", ARTIFACT_DIR)
    yield


app = FastAPI(
    title="Bitcoin Transaction Forensics API",
    description="Orchestrated ML forensic pipeline with graph, fusion scoring, and SHAP explainability.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "Bitcoin Transaction Forensics API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "upload": "POST /api/v1/jobs/upload",
    }


app.include_router(jobs.router)
app.include_router(graph.router)
app.include_router(alerts.router)
app.include_router(explainability.router)
app.include_router(llm_explain.router)
app.include_router(patterns.router)
app.include_router(health.router)


def create_app() -> FastAPI:
    """Return the application instance (used by tests and ASGI servers)."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
