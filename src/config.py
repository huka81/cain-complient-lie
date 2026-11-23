from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Load .env from repository root if present
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")


class ExperimentConfig(BaseModel):
    """Single source of experiment parameters (overridable via env vars or .env)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # LLM Provider: 'nvidia' or 'ollama'
    llm_provider: str = Field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "nvidia"))
    default_model: str = Field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    )

    # NVIDIA API configuration
    nvidia_api_key: str = Field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY", ""))
    nvidia_invoke_url: str = Field(
        default_factory=lambda: os.environ.get(
            "NVIDIA_INVOKE_URL", "https://integrate.api.nvidia.com/v1/chat/completions"
        )
    )
    nvidia_temperature: float = Field(
        default_factory=lambda: float(os.environ.get("NVIDIA_TEMPERATURE", "0.0"))
    )
    nvidia_max_tokens: int = Field(
        default_factory=lambda: int(os.environ.get("NVIDIA_MAX_TOKENS", "1024"))
    )
    nvidia_top_p: float = Field(
        default_factory=lambda: float(os.environ.get("NVIDIA_TOP_P", "1.0"))
    )
    nvidia_seed: int = Field(
        default_factory=lambda: int(os.environ.get("NVIDIA_SEED", "0"))
    )
    nvidia_stream: bool = Field(
        default_factory=lambda: os.environ.get("NVIDIA_STREAM", "false").lower() in ("true", "1", "yes")
    )
    nvidia_enable_thinking: bool = Field(
        default_factory=lambda: os.environ.get("NVIDIA_ENABLE_THINKING", "false").lower() in ("true", "1", "yes")
    )

    # NVIDIA error handling / retries
    nvidia_max_retries: int = Field(
        default_factory=lambda: int(os.environ.get("NVIDIA_MAX_RETRIES", "6")),
        description="Max retries for transient errors (429/5xx/timeout). Permanent 4xx errors fail fast.",
    )
    nvidia_backoff_base_s: float = Field(
        default_factory=lambda: float(os.environ.get("NVIDIA_BACKOFF_BASE_S", "1.0")),
        description="Base for exponential backoff: base * 2**attempt, plus jitter, capped at backoff_cap_s.",
    )
    nvidia_backoff_cap_s: float = Field(
        default_factory=lambda: float(os.environ.get("NVIDIA_BACKOFF_CAP_S", "30.0")),
        description="Upper bound on a single backoff sleep.",
    )
    nvidia_respect_retry_after: bool = Field(
        default_factory=lambda: os.environ.get("NVIDIA_RESPECT_RETRY_AFTER", "true").lower() in ("true", "1", "yes"),
        description="Honor the Retry-After header on 429/503 instead of computed backoff.",
    )
    nvidia_adaptive_rate: bool = Field(
        default_factory=lambda: os.environ.get("NVIDIA_ADAPTIVE_RATE", "true").lower() in ("true", "1", "yes"),
        description="AIMD throttle: widen request interval after 429, relax after sustained successes.",
    )

    # NVIDIA model discovery (scripts/scan_nvidia_models.py)
    nvidia_discovery_timeout_s: float = Field(
        default_factory=lambda: float(os.environ.get("NVIDIA_DISCOVERY_TIMEOUT_S", "5.0")),
        description="Per-model probe timeout when scanning for active models.",
    )
    nvidia_discovery_skip_specialized: bool = Field(
        default_factory=lambda: os.environ.get("NVIDIA_DISCOVERY_SKIP_SPECIALIZED", "true").lower()
        in ("true", "1", "yes"),
        description="Skip embedding/rerank/audio/image/guard models when probing chat availability.",
    )
    nvidia_available_models: tuple[str, ...] = (
        "nvidia/nemotron-3.5-bolt-e4b",
        "poolside/laguna-xs-2.1",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "mistralai/mistral-large-4-1t-a52b",
    )

    # Ollama configuration
    # 127.0.0.1, not localhost: Windows IPv6-first resolution adds ~21 s per request.
    ollama_url: str = Field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    slider_models: tuple[str, ...] = ("google/gemma-4-31b-it",)
    max_workers: int = Field(default_factory=lambda: int(os.environ.get("LLM_MAX_WORKERS", "2")))
    cache_filename: str = "nvidia_llm_cache.sqlite"
    min_request_interval_s: float = Field(default_factory=lambda: float(os.environ.get("LLM_MIN_INTERVAL_S", "4.0")))
    temperature: float = 0.0
    seed: int = 42
    num_predict: int = 256
    request_timeout_s: float = 600.0

    dataset_seed: int = 42
    test_seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    dev_seed: int = 0
    cases_per_class_per_scenario_per_seed: int = 4
    dev_cases_per_class_per_scenario: int = 20
    cases_per_class_full: int = 30
    cases_per_class_fast: int = 10
    cascade_flows: int = 1000

    c_human_eur: float = Field(default=5.0, description="Cost of one human escalation review.")
    l_biz_eur: float = Field(default=80.0, description="Expected business loss of one undetected Compliant Lie.")
    eur_per_million_tokens: dict[str, float] = Field(
        default_factory=lambda: {"qwen3:1.7b": 0.03, "llama3.2:3b": 0.06, "gemma3:4b": 0.10, "google/gemma-4-31b-it": 0.30},
        description="Hosted-inference-equivalent price per 1M tokens for each local model tier.",
    )

    tau_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    tau_fine_step: float = 0.05
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 7

    def price_per_token(self, model: str) -> float:
        return self.eur_per_million_tokens.get(model, 0.06) / 1_000_000.0
