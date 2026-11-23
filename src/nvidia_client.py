from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from collections import deque
from typing import Any, TypeVar

import requests
from pydantic import BaseModel, ConfigDict, ValidationError

from config import ExperimentConfig
from llm_cache import CachedResponse, LLMCache, request_key

logger: logging.Logger = logging.getLogger("nvidia")

T = TypeVar("T", bound=BaseModel)

_THINK_RE: re.Pattern[str] = re.compile(r"<think>.*?</think>", re.S)

# HTTP statuses that are transient (worth retrying) vs permanent (fail fast).
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_PERMANENT_STATUS: frozenset[int] = frozenset({400, 401, 403, 404, 410})


class NvidiaError(RuntimeError):
    """Base class for NVIDIA NIM client errors."""


# Backward-compatible alias: existing imports/tests use NvidiaUnavailable as the base.
NvidiaUnavailable = NvidiaError


class NvidiaRateLimited(NvidiaError):
    """HTTP 429 - request quota / rate limit exceeded (transient)."""


class NvidiaModelUnavailable(NvidiaError):
    """HTTP 404/410/403 - model not found, retired (EOL) or not entitled (permanent)."""


class NvidiaServerError(NvidiaError):
    """HTTP 5xx - server-side error (transient)."""


class NvidiaTimeout(NvidiaError):
    """Network timeout or transport failure (transient)."""


class CallStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool
    parse_error: bool


class UsageStats(BaseModel):
    """Aggregated usage counters for a client instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    live_calls: int  # successful, non-cached chat completions
    cached_calls: int  # served from the SQLite cache
    http_requests: int  # total HTTP POSTs sent (incl. retries) - what the rate limit sees
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    requests_last_minute: int  # HTTP POSTs in the trailing 60 s window
    peak_requests_per_minute: int  # highest observed trailing-60 s count


class NvidiaClient:
    """Structured-output NVIDIA NIM chat client with SQLite caching, adaptive
    rate limiting and classifying retry/error handling."""

    def __init__(self, config: ExperimentConfig, cache: LLMCache) -> None:
        self._cfg = config
        self._cache = cache
        self.live_calls: int = 0
        self.cached_calls: int = 0
        self.http_requests: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self._peak_rpm: int = 0
        self._request_times: deque[float] = deque()
        self._count_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._next_slot = 0.0
        # Adaptive (AIMD) request interval: starts at the configured floor, widens on 429.
        self._interval = float(config.min_request_interval_s)
        self._ok_streak = 0

    def _throttle(self) -> None:
        """Space request starts by the current adaptive interval across worker threads;
        the hosted endpoint returns 429 otherwise."""
        with self._rate_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait:
            time.sleep(wait)

    def _record_http_request(self) -> None:
        """Record one outgoing HTTP POST timestamp and update the trailing-minute rate."""
        now = time.monotonic()
        with self._count_lock:
            self.http_requests += 1
            self._request_times.append(now)
            cutoff = now - 60.0
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()
            rpm = len(self._request_times)
            if rpm > self._peak_rpm:
                self._peak_rpm = rpm

    def requests_last_minute(self) -> int:
        """HTTP POSTs sent in the trailing 60 s window."""
        now = time.monotonic()
        cutoff = now - 60.0
        with self._count_lock:
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()
            return len(self._request_times)

    def usage_stats(self) -> UsageStats:
        """Snapshot of cumulative token and request counters for this client."""
        rpm = self.requests_last_minute()
        with self._count_lock:
            return UsageStats(
                live_calls=self.live_calls,
                cached_calls=self.cached_calls,
                http_requests=self.http_requests,
                prompt_tokens=self.total_prompt_tokens,
                completion_tokens=self.total_completion_tokens,
                total_tokens=self.total_prompt_tokens + self.total_completion_tokens,
                requests_last_minute=rpm,
                peak_requests_per_minute=self._peak_rpm,
            )

    def _on_rate_limited(self) -> None:
        """AIMD multiplicative increase: double the interval (capped) after a 429."""
        if not self._cfg.nvidia_adaptive_rate:
            return
        with self._rate_lock:
            self._ok_streak = 0
            self._interval = min(self._interval * 2.0, self._cfg.nvidia_backoff_cap_s)
        logger.warning("nvidia_rate_adapt interval=%.2fs", self._interval)

    def _on_success(self) -> None:
        """AIMD additive decrease: relax the interval toward the floor after sustained successes."""
        if not self._cfg.nvidia_adaptive_rate:
            return
        with self._rate_lock:
            self._ok_streak += 1
            if self._ok_streak >= 5 and self._interval > self._cfg.min_request_interval_s:
                self._interval = max(self._cfg.min_request_interval_s, self._interval - self._cfg.min_request_interval_s)
                self._ok_streak = 0

    def _backoff_sleep(self, attempt: int, retry_after: float | None) -> None:
        """Sleep before the next retry: honor Retry-After when present, else exponential backoff + jitter."""
        if retry_after is not None and self._cfg.nvidia_respect_retry_after:
            delay = min(retry_after, self._cfg.nvidia_backoff_cap_s)
        else:
            base = self._cfg.nvidia_backoff_base_s * (2.0**attempt)
            delay = min(base, self._cfg.nvidia_backoff_cap_s)
            delay += random.uniform(0.0, self._cfg.nvidia_backoff_base_s)
        time.sleep(delay)

    @staticmethod
    def _parse_retry_after(resp: requests.Response) -> float | None:
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def options(self) -> dict[str, Any]:
        return {
            "temperature": self._cfg.nvidia_temperature,
            "max_tokens": self._cfg.nvidia_max_tokens,
            "top_p": self._cfg.nvidia_top_p,
            "seed": self._cfg.nvidia_seed,
            "enable_thinking": self._cfg.nvidia_enable_thinking,
        }

    def check_connectivity(self, required_models: tuple[str, ...]) -> list[str]:
        headers = {
            "Authorization": f"Bearer {self._cfg.nvidia_api_key}",
            "Accept": "application/json",
        }
        base_url = self._cfg.nvidia_invoke_url.rsplit("/chat/completions", 1)[0]
        try:
            resp = requests.get(f"{base_url}/models", headers=headers, timeout=10)
            if resp.status_code != 200:
                raise NvidiaUnavailable(f"NVIDIA API returned status {resp.status_code}: {resp.text}")
            data = resp.json()
        except Exception as exc:
            raise NvidiaUnavailable(f"Cannot reach NVIDIA API at {base_url}") from exc
        available = {m["id"] for m in data.get("data", [])}
        missing = [m for m in required_models if m not in available]
        if missing:
            logger.warning("Missing NVIDIA models from /models listing: %s (may still be callable)", missing)
        return sorted(available)

    def chat_structured(self, model: str, system: str, user: str, schema_model: type[T]) -> tuple[T | None, CallStats]:
        schema = schema_model.model_json_schema()
        opts = self.options()
        key = request_key(f"nvidia:{model}", system, user, schema, opts)
        cached = self._cache.get(key)
        hit = cached is not None
        if cached is None:
            cached = self._post(model, system, user, schema, opts)
            self._cache.put(key, model, cached)
            with self._count_lock:
                self.live_calls += 1
                self.total_prompt_tokens += cached.prompt_tokens
                self.total_completion_tokens += cached.completion_tokens
        else:
            with self._count_lock:
                self.cached_calls += 1
        content = _THINK_RE.sub("", cached.content).strip()
        parsed = self._parse_json(content, schema_model)
        if parsed is None:
            logger.warning("structured_output_invalid model=%s content=%r", model, content[:200])
        stats = CallStats(
            prompt_tokens=cached.prompt_tokens,
            completion_tokens=cached.completion_tokens,
            latency_s=cached.latency_s,
            cached=hit,
            parse_error=parsed is None,
        )
        return parsed, stats

    def _parse_json(self, content: str, schema_model: type[T]) -> T | None:
        try:
            return schema_model.model_validate_json(content)
        except ValidationError:
            pass

        cleaned = content
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            return schema_model.model_validate_json(cleaned)
        except (ValidationError, ValueError):
            pass

        m = re.search(r"(\{.*\})", cleaned, re.S)
        if m:
            try:
                return schema_model.model_validate_json(m.group(1))
            except (ValidationError, ValueError):
                pass
        return None

    def _post(
        self, model: str, system: str, user: str, schema: dict[str, Any], opts: dict[str, Any]
    ) -> CachedResponse:
        schema_instruction = (
            f"\n\nYou MUST respond ONLY with a JSON object that strictly adheres to the following JSON Schema. "
            f"Do not enclose the JSON in markdown code blocks. Do not output any explanation or extra text.\n"
            f"Schema:\n{json.dumps(schema)}"
        )
        messages = [
            {"role": "system", "content": system + schema_instruction},
            {"role": "user", "content": user},
        ]
        headers = {
            "Authorization": f"Bearer {self._cfg.nvidia_api_key}",
            "Accept": "text/event-stream" if self._cfg.nvidia_stream else "application/json",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": opts.get("max_tokens", self._cfg.nvidia_max_tokens),
            "temperature": opts.get("temperature", self._cfg.nvidia_temperature),
            "top_p": opts.get("top_p", self._cfg.nvidia_top_p),
            "seed": opts.get("seed", self._cfg.nvidia_seed),
            "stream": self._cfg.nvidia_stream,
            "chat_template_kwargs": {"enable_thinking": bool(opts.get("enable_thinking", False))},
        }
        last_exc: Exception | None = None
        for attempt in range(self._cfg.nvidia_max_retries + 1):
            self._throttle()
            self._record_http_request()
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    self._cfg.nvidia_invoke_url,
                    headers=headers,
                    json=payload,
                    timeout=min(self._cfg.request_timeout_s, 120.0),
                    stream=self._cfg.nvidia_stream,
                )
            except (requests.RequestException, TimeoutError) as exc:
                last_exc = NvidiaTimeout(f"NVIDIA transport error for {model}: {exc}")
                logger.warning("nvidia_retry attempt=%d err=%s", attempt + 1, exc)
                if attempt < self._cfg.nvidia_max_retries:
                    self._backoff_sleep(attempt, None)
                continue

            latency = time.perf_counter() - t0
            status = resp.status_code
            if status == 200:
                self._on_success()
                return self._read_success(resp, latency)

            if status in _PERMANENT_STATUS:
                # Non-retryable: fail fast with a precise exception.
                detail = f"NVIDIA API error ({status}) for {model}: {resp.text[:200]}"
                if status in (403, 404, 410):
                    raise NvidiaModelUnavailable(detail)
                raise NvidiaError(detail)

            if status in _RETRYABLE_STATUS:
                retry_after = self._parse_retry_after(resp)
                if status == 429:
                    self._on_rate_limited()
                    last_exc = NvidiaRateLimited(f"NVIDIA rate limited (429) for {model}")
                else:
                    last_exc = NvidiaServerError(f"NVIDIA server error ({status}) for {model}")
                logger.warning(
                    "nvidia_retry attempt=%d status=%d retry_after=%s text=%s",
                    attempt + 1,
                    status,
                    retry_after,
                    resp.text[:200],
                )
                if attempt < self._cfg.nvidia_max_retries:
                    self._backoff_sleep(attempt, retry_after)
                continue

            # Unknown status: treat as a hard error.
            raise NvidiaError(f"NVIDIA API unexpected status ({status}) for {model}: {resp.text[:200]}")

        raise (last_exc or NvidiaError(f"NVIDIA call failed for {model}"))

    def _read_success(self, resp: requests.Response, latency: float) -> CachedResponse:
        if self._cfg.nvidia_stream:
            content_chunks: list[str] = []
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        piece = delta.get("content") or delta.get("reasoning_content") or ""
                        if piece:
                            content_chunks.append(piece)
                    except Exception:
                        pass
            full_content = "".join(content_chunks)
            return CachedResponse(
                content=full_content,
                prompt_tokens=0,
                completion_tokens=len(full_content.split()),
                latency_s=latency,
            )

        data = resp.json()
        usage = data.get("usage", {})
        choices = data.get("choices", [])
        msg = choices[0].get("message", {}) if choices else {}
        # Reasoning models (e.g. nemotron) may return content=None with the answer in reasoning_content.
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return CachedResponse(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_s=latency,
        )


def main() -> None:
    """Scan NVIDIA NIM for actively available chat models and print them."""
    from nvidia_models import find_working_models

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = ExperimentConfig()
    if not cfg.nvidia_api_key:
        print("NVIDIA_API_KEY is not set (configure it in .env).")
        raise SystemExit(1)

    print("Scanning NVIDIA NIM for available chat models...")
    result = find_working_models(cfg)

    print(f"\nAvailable models ({len(result.working)}):")
    for model_id in result.working:
        print(f"  + {model_id}")

    if result.rate_limited:
        print(f"\nRate-limited / in-plan ({len(result.rate_limited)}):")
        for model_id in result.rate_limited:
            print(f"  * {model_id}")


if __name__ == "__main__":
    main()

