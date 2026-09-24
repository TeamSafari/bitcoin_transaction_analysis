"""Tests for LLM explanation caching in SQLite database."""

from __future__ import annotations

import tempfile
from pathlib import Path

from backend.database import (
    delete_wallet_explanations,
    get_wallet_explanation,
    init_db,
    save_wallet_explanation,
)


def test_wallet_explanation_caching():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_jobs.db"
        init_db(db_path)

        # 1. Initially no explanation
        assert get_wallet_explanation("job-1", "W001", db_path=db_path) is None

        # 2. Save explanation
        save_wallet_explanation(
            job_id="job-1",
            wallet_id="W001",
            explanation="Suspicious peel chain detected.",
            model_used="Qwen2.5-1.5B-Instruct",
            db_path=db_path,
        )

        # 3. Retrieve explanation
        cached = get_wallet_explanation("job-1", "W001", db_path=db_path)
        assert cached is not None
        assert cached["wallet_id"] == "W001"
        assert cached["explanation"] == "Suspicious peel chain detected."
        assert cached["model_used"] == "Qwen2.5-1.5B-Instruct"
        assert cached["cached"] is True

        # 4. Overwrite explanation on conflict
        save_wallet_explanation(
            job_id="job-1",
            wallet_id="W001",
            explanation="Updated explanation text.",
            model_used="Qwen2.5-1.5B-Instruct",
            db_path=db_path,
        )
        updated = get_wallet_explanation("job-1", "W001", db_path=db_path)
        assert updated is not None
        assert updated["explanation"] == "Updated explanation text."

        # 5. Different job or wallet returns None
        assert get_wallet_explanation("job-2", "W001", db_path=db_path) is None
        assert get_wallet_explanation("job-1", "W002", db_path=db_path) is None

        # 6. Delete explanations for a job
        delete_wallet_explanations("job-1", db_path=db_path)
        assert get_wallet_explanation("job-1", "W001", db_path=db_path) is None
