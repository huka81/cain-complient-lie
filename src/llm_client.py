from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from config import ExperimentConfig
from llm_cache import LLMCache
from ollama_client import CallStats

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """Protocol for LLM chat clients that return structured Pydantic models."""

    def chat_structured(
        self, model: str, system: str, user: str, schema_model: type[T]
    ) -> tuple[T | None, CallStats]: ...


def get_llm_client(config: ExperimentConfig, cache: LLMCache) -> LLMClient:
    """Factory to instantiate the configured LLM client."""
    provider = config.llm_provider.lower().strip()
    if provider == "nvidia":
        from nvidia_client import NvidiaClient

        return NvidiaClient(config, cache)
    elif provider == "ollama":
        from ollama_client import OllamaClient

        return OllamaClient(config, cache)
    raise ValueError(f"Unknown LLM provider: {config.llm_provider}. Supported providers: 'nvidia', 'ollama'")
