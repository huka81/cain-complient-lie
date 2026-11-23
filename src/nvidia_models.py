"""Active-model discovery for NVIDIA NIM (requests-based).

Lists models from the hosted catalog and actively probes each one with a tiny
chat request to classify real availability (available / rate-limited / EOL /
not-found / forbidden). Mirrors the behaviour of the former ``nvidia_llm.py``
reference while reusing this project's :class:`ExperimentConfig`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import requests

from config import ExperimentConfig

logger: logging.Logger = logging.getLogger("nvidia.models")

# Curated list of chat models verified as active on NVIDIA NIM, ordered by speed.
AVAILABLE_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3.5-bolt-e4b",
    "poolside/laguna-xs-2.1",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "mistralai/mistral-large-4-1t-a52b",
)

# Substrings marking non-chat / specialized models skipped during probing.
SPECIALIZED_MARKERS: tuple[str, ...] = (
    "embed",
    "rerank",
    "clip",
    "whisper",
    "audio",
    "tts",
    "sdxl",
    "flux",
    "guard",
    "translate",
)


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"  # 429 (in plan, quota momentarily exhausted)
    EOL = "eol"  # 410 Gone
    NOT_FOUND = "not_found"  # 404
    FORBIDDEN = "forbidden"  # 401 / 403
    ERROR = "error"  # anything else


@dataclass
class DiscoveryResult:
    """Classified outcome of scanning the NVIDIA NIM catalog."""

    scanned: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    working: list[str] = field(default_factory=list)
    rate_limited: list[str] = field(default_factory=list)
    eol: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    probe_requests: int = 0  # number of probe HTTP POSTs sent
    elapsed_s: float = 0.0  # wall-clock duration of the probing loop

    @property
    def requests_per_minute(self) -> float:
        """Effective probe throughput (requests per minute) during the scan."""
        if self.elapsed_s <= 0:
            return 0.0
        return self.probe_requests * 60.0 / self.elapsed_s


def _base_url(cfg: ExperimentConfig) -> str:
    return cfg.nvidia_invoke_url.rsplit("/chat/completions", 1)[0]


def _auth_headers(cfg: ExperimentConfig) -> dict[str, str]:
    if not cfg.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is not set")
    return {"Authorization": f"Bearer {cfg.nvidia_api_key}", "Accept": "application/json"}


def is_specialized(model_id: str) -> bool:
    mid = model_id.lower()
    return any(marker in mid for marker in SPECIALIZED_MARKERS)


def list_remote_models(cfg: ExperimentConfig, timeout: float = 10.0) -> list[str]:
    """Return every model id advertised by the hosted catalog (GET /v1/models, free)."""
    resp = requests.get(f"{_base_url(cfg)}/models", headers=_auth_headers(cfg), timeout=timeout)
    if resp.status_code != 200:
        raise requests.HTTPError(f"NVIDIA /models returned {resp.status_code}: {resp.text[:200]}")
    return [m["id"] for m in resp.json().get("data", [])]


def classify_status(status_code: int) -> ModelStatus:
    if status_code == 200:
        return ModelStatus.AVAILABLE
    if status_code == 429:
        return ModelStatus.RATE_LIMITED
    if status_code == 410:
        return ModelStatus.EOL
    if status_code == 404:
        return ModelStatus.NOT_FOUND
    if status_code in (401, 403):
        return ModelStatus.FORBIDDEN
    return ModelStatus.ERROR


def probe_model(cfg: ExperimentConfig, model_id: str, timeout: float | None = None) -> ModelStatus:
    """Send a minimal chat request to classify a single model's real availability."""
    timeout = timeout if timeout is not None else cfg.nvidia_discovery_timeout_s
    headers = {**_auth_headers(cfg), "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        resp = requests.post(cfg.nvidia_invoke_url, headers=headers, json=payload, timeout=timeout)
    except (requests.RequestException, TimeoutError) as exc:
        logger.debug("probe_error model=%s err=%s", model_id, exc)
        return ModelStatus.ERROR
    return classify_status(resp.status_code)


def find_working_models(
    cfg: ExperimentConfig,
    *,
    skip_specialized: bool | None = None,
    timeout: float | None = None,
) -> DiscoveryResult:
    """List the catalog and actively probe each chat model, classifying availability."""
    skip_specialized = cfg.nvidia_discovery_skip_specialized if skip_specialized is None else skip_specialized
    timeout = timeout if timeout is not None else cfg.nvidia_discovery_timeout_s

    result = DiscoveryResult()
    model_ids = list_remote_models(cfg)
    logger.info("nvidia_discovery catalog=%d skip_specialized=%s", len(model_ids), skip_specialized)

    t0 = time.monotonic()
    for model_id in model_ids:
        if skip_specialized and is_specialized(model_id):
            result.skipped.append(model_id)
            continue
        result.scanned.append(model_id)
        status = probe_model(cfg, model_id, timeout)
        result.probe_requests += 1
        if status is ModelStatus.AVAILABLE:
            result.working.append(model_id)
        elif status is ModelStatus.RATE_LIMITED:
            result.rate_limited.append(model_id)
        elif status is ModelStatus.EOL:
            result.eol.append(model_id)
        elif status is ModelStatus.NOT_FOUND:
            result.not_found.append(model_id)
        elif status is ModelStatus.FORBIDDEN:
            result.forbidden.append(model_id)
        else:
            result.errors.append((model_id, status.value))
    result.elapsed_s = time.monotonic() - t0

    return result
