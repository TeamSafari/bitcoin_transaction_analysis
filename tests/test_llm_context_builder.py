"""Tests for LLM context builder and prompt assembly."""

from __future__ import annotations

from pathlib import Path
import pytest

from backend.api.services.data_loader import DataLoader
from backend.llm_agent.config import SHAP_EXCLUDE_PREFIXES
from backend.llm_agent.context_builder import (
    _format_feature_value,
    _format_sats,
    _sats_to_btc,
    _seconds_to_human,
    build_context,
)


def test_formatting_helpers():
    # sats to btc helper (legacy/utility)
    assert "1.5000 BTC" in _sats_to_btc(150000000)
    assert "0.005000 BTC" in _sats_to_btc(500000)
    assert "50 sats" in _sats_to_btc(50)

    # format sats
    assert _format_sats(150000000) == "150000000 sats"
    assert _format_sats(50) == "50 sats"
    assert _format_feature_value("median_sent_sats", 13379910.0) == "13379910 sats"

    # seconds to duration
    assert "2.0 days" in _seconds_to_human(172800)
    assert "5.0 hours" in _seconds_to_human(18000)
    assert "45 seconds" in _seconds_to_human(45)


def test_build_context_with_job():
    job_results_dir = Path("outputs/jobs/job-0c49abe5/results")
    if not job_results_dir.exists():
        pytest.skip("Test pipeline output job-0c49abe5 results directory not present")

    loader = DataLoader(job_results_dir)
    sys_prompt, user_prompt = build_context("W0000182", loader)

    assert "cryptocurrency forensics analyst" in sys_prompt
    assert "sats" in sys_prompt
    assert "do NOT use BTC" in sys_prompt
    assert "WALLET: W0000182" in user_prompt
    assert "TRANSACTION SATS: input_sats=" in user_prompt
    assert "output_sats=" in user_prompt
    assert "367836687 sats" in user_prompt
    assert "368616596 sats" in user_prompt
    assert "RISK VERDICT:" in user_prompt
    assert "TOP CONTRIBUTING FEATURES" in user_prompt
    assert "BEHAVIORAL PROFILE:" in user_prompt
    assert "ANOMALY SIGNALS:" in user_prompt

    # Ensure excluded prefixes like graphsage_dim_ are not in the prompt
    for prefix in SHAP_EXCLUDE_PREFIXES:
        assert prefix not in user_prompt
