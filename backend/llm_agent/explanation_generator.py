"""
Orchestrates the full explanation flow:
  1. Check SQLite cache
  2. Build prompt from pipeline outputs
  3. Run LLM inference
  4. Cache result in SQLite
  5. Return explanation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from backend.api.services.data_loader import DataLoader
from backend.database import get_wallet_explanation, save_wallet_explanation
from backend.llm_agent.context_builder import build_context
from backend.llm_agent.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generate and cache natural language wallet explanations."""

    def __init__(
        self,
        loader: DataLoader,
        job_id: Optional[str] = None,
    ) -> None:
        self.loader = loader
        self.job_id = job_id

    def explain(self, wallet_id: str) -> Dict[str, Any]:
        """
        Generate or retrieve a cached explanation for a wallet.

        Returns a dict with keys: wallet_id, explanation, model_used, cached.
        """
        effective_job_id = self.job_id or "__default__"

        # 1. Check cache
        cached = get_wallet_explanation(effective_job_id, wallet_id)
        if cached is not None:
            logger.info("Cache hit for explanation: job=%s wallet=%s", effective_job_id, wallet_id)
            cached["cached"] = True
            return cached

        # 2. Build prompt
        logger.info("Generating explanation: job=%s wallet=%s", effective_job_id, wallet_id)
        system_prompt, user_prompt = build_context(wallet_id, self.loader)

        # 3. Run inference
        model = ModelManager.get_instance()
        explanation = model.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        # 4. Build result
        result = {
            "wallet_id": wallet_id,
            "explanation": explanation,
            "model_used": "Qwen2.5-1.5B-Instruct",
            "cached": False,
        }

        # 5. Cache in SQLite
        save_wallet_explanation(
            job_id=effective_job_id,
            wallet_id=wallet_id,
            explanation=explanation,
            model_used="Qwen2.5-1.5B-Instruct",
        )

        return result
