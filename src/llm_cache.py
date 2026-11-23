from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class CachedResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


def request_key(model: str, system: str, user: str, schema: dict[str, Any], options: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "user": user, "schema": schema, "options": options},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    """SQLite response cache keyed by SHA256(model + prompt + schema + options)."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            " key TEXT PRIMARY KEY, model TEXT NOT NULL, content TEXT NOT NULL,"
            " prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL,"
            " latency_s REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> CachedResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT content, prompt_tokens, completion_tokens, latency_s FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return CachedResponse(content=row[0], prompt_tokens=row[1], completion_tokens=row[2], latency_s=row[3])

    def put(self, key: str, model: str, response: CachedResponse) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?)",
                (key, model, response.content, response.prompt_tokens, response.completion_tokens, response.latency_s),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
