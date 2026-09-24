"""API tests for the LLM explanation endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.database import save_wallet_explanation

client = TestClient(create_app())


def test_explain_wallet_cached():
    # Pre-populate cache in SQLite
    save_wallet_explanation(
        job_id="job-0c49abe5",
        wallet_id="W0000182",
        explanation="Cached explanation from automated test.",
        model_used="Qwen2.5-1.5B-Instruct",
    )

    response = client.get("/api/v1/jobs/job-0c49abe5/explain/wallet/W0000182")
    assert response.status_code == 200
    data = response.json()
    assert data["wallet_id"] == "W0000182"
    assert data["explanation"] == "Cached explanation from automated test."
    assert data["cached"] is True
    assert data["model_used"] == "Qwen2.5-1.5B-Instruct"


def test_explain_wallet_nonexistent_job():
    response = client.get("/api/v1/jobs/nonexistent-job-xyz/explain/wallet/W0000001")
    assert response.status_code == 404
