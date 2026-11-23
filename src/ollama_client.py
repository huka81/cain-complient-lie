from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from config import ExperimentConfig
from llm_cache import CachedResponse, LLMCache, request_key

logger: logging.Logger = logging.getLogger("ollama")

T = TypeVar("T", bound=BaseModel)

_THINK_RE: re.Pattern[str] = re.compile(r"<think>.*?</think>", re.S)


class OllamaUnavailable(RuntimeError):
    """Ollama endpoint unreachable or returned a transport-level error."""


class CallStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool
    parse_error: bool


class OllamaClient:
    """Structured-output Ollama chat client with deterministic options and SQLite caching."""

    def __init__(self, config: ExperimentConfig, cache: LLMCache) -> None:
        self._cfg = config
        self._cache = cache
        self.live_calls: int = 0

    def options(self) -> dict[str, Any]:
        return {"temperature": self._cfg.temperature, "seed": self._cfg.seed, "num_predict": self._cfg.num_predict}

    def check_connectivity(self, required_models: tuple[str, ...]) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self._cfg.ollama_url}/api/tags", timeout=10) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaUnavailable(f"Cannot reach Ollama at {self._cfg.ollama_url}") from exc
        available = {m["name"] for m in tags.get("models", [])}
        missing = [m for m in required_models if m not in available]
        if missing:
            raise OllamaUnavailable(f"Missing Ollama models: {missing}")
        return sorted(available)

    def chat_structured(self, model: str, system: str, user: str, schema_model: type[T]) -> tuple[T | None, CallStats]:
        schema = schema_model.model_json_schema()
        is_qwen3 = model.startswith("qwen3")
        if is_qwen3:
            user = f"{user}\n/no_think"
        opts = self.options()
        key = request_key(model, system, user, schema, opts)
        cached = self._cache.get(key)
        hit = cached is not None
        if cached is None:
            cached = self._post(model, system, user, schema, opts, think_off=is_qwen3)
            self._cache.put(key, model, cached)
            self.live_calls += 1
        content = _THINK_RE.sub("", cached.content).strip()
        parsed: T | None
        try:
            parsed = schema_model.model_validate_json(content)
        except ValidationError:
            logger.warning("structured_output_invalid model=%s content=%r", model, content[:200])
            parsed = None
        stats = CallStats(
            prompt_tokens=cached.prompt_tokens,
            completion_tokens=cached.completion_tokens,
            latency_s=cached.latency_s,
            cached=hit,
            parse_error=parsed is None,
        )
        return parsed, stats

    def _post(
        self, model: str, system: str, user: str, schema: dict[str, Any], opts: dict[str, Any], *, think_off: bool
    ) -> CachedResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": opts,
            "keep_alive": "30m",
        }
        if think_off:
            body["think"] = False
        data = json.dumps(body).encode("utf-8")
        last_exc: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(
                f"{self._cfg.ollama_url}/api/chat", data=data, headers={"Content-Type": "application/json"}
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self._cfg.request_timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_exc = exc
                logger.warning("ollama_retry attempt=%d model=%s err=%s", attempt + 1, model, exc)
                time.sleep(2.0 * (attempt + 1))
                continue
            latency = time.perf_counter() - t0
            return CachedResponse(
                content=payload.get("message", {}).get("content", ""),
                prompt_tokens=int(payload.get("prompt_eval_count", 0) or 0),
                completion_tokens=int(payload.get("eval_count", 0) or 0),
                latency_s=latency,
            )
        raise OllamaUnavailable(f"Ollama call failed for {model}") from last_exc
