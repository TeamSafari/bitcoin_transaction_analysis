"""
Singleton manager for the local GGUF LLM model.

Loads the model once and reuses across requests. Thread-safe.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from backend.llm_agent.config import (
    LLM_CONTEXT_SIZE,
    LLM_MAX_TOKENS,
    LLM_MODEL_DIR,
    LLM_MODEL_FILE,
    LLM_N_THREADS,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe singleton for llama-cpp-python inference."""

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._model_path: Path = LLM_MODEL_DIR / LLM_MODEL_FILE
        self._inference_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def model_available(self) -> bool:
        """Check whether the GGUF model file exists on disk."""
        return self._model_path.exists()

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first inference call."""
        if self._model is not None:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"LLM model not found at {self._model_path}. "
                f"Please download the GGUF file and place it at: {self._model_path}"
            )

        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python>=0.3.0"
            ) from exc

        logger.info("Loading LLM model from %s ...", self._model_path)
        self._model = Llama(
            model_path=str(self._model_path),
            n_ctx=LLM_CONTEXT_SIZE,
            n_threads=LLM_N_THREADS,
            verbose=False,
        )
        logger.info("LLM model loaded successfully.")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        stop: list[str] | None = None,
    ) -> str:
        """
        Run inference and return the generated text.

        Uses chat completion format for instruct models.
        Thread-safe: only one inference runs at a time.
        """
        with self._inference_lock:
            self._ensure_loaded()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
            )

            choice = response["choices"][0]  # type: ignore[index]
            return choice["message"]["content"].strip()
