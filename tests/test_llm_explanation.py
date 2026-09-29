"""End-to-end tests for the ExplanationGenerator with mock model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.api.services.data_loader import DataLoader
from backend.database import delete_wallet_explanations, get_wallet_explanation
from backend.llm_agent.explanation_generator import ExplanationGenerator
from backend.llm_agent.model_manager import ModelManager


def test_explanation_generator_flow():
    job_results_dir = Path("outputs/jobs/job-0c49abe5/results")
    if not job_results_dir.exists():
        pytest.skip("Test pipeline output job-0c49abe5 results directory not present")

    # Clear any leftover cached explanation for the mock test job
    delete_wallet_explanations("test-job-mock")

    loader = DataLoader(job_results_dir)
    generator = ExplanationGenerator(loader, job_id="test-job-mock")

    try:
        # Mock ModelManager.generate method so we don't load the full model
        with patch.object(ModelManager, "generate", return_value="Mocked LLM explanation for W0000182."):
            # First call: fresh generation
            result = generator.explain("W0000182")
            assert result["wallet_id"] == "W0000182"
            assert result["explanation"] == "Mocked LLM explanation for W0000182."
            assert result["cached"] is False
            assert result["model_used"] == "Qwen2.5-1.5B-Instruct"


        # Verify it was stored in SQLite
        db_record = get_wallet_explanation("test-job-mock", "W0000182")
        assert db_record is not None
        assert db_record["explanation"] == "Mocked LLM explanation for W0000182."

        # Second call: should come from SQLite cache without calling model.generate
        with patch.object(ModelManager, "generate") as mock_gen:
            cached_result = generator.explain("W0000182")
            assert cached_result["wallet_id"] == "W0000182"
            assert cached_result["explanation"] == "Mocked LLM explanation for W0000182."
            assert cached_result["cached"] is True
            mock_gen.assert_not_called()
    finally:
        delete_wallet_explanations("test-job-mock")

