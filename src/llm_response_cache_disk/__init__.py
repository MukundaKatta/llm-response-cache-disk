"""
llm-response-cache-disk: SQLite-backed disk cache for LLM responses.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Optional


def _hash_request(messages: list[dict[str, Any]], model: str = "", **extras: Any) -> str:
    payload = {"messages": messages, "model": model, **extras}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class CacheEntry:
    key: str
    response: Any
    created_at: float
    ttl: Optional[float]
    hit_count: int = 0
    model: str = ""

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) >= self.ttl


class DiskResponseCache:
    """
    SQLite-backed disk cache for LLM responses. Persists across restarts.

    Usage::

        cache = DiskResponseCache("/tmp/llm-cache.db", default_ttl=3600)
        key = cache.put(messages, response, model="claude")
        cached = cache.get(messages, model="claude")
    """

    def __init__(self, db_path: str, default_ttl: Optional[float] = None) -> None:
        self._db_path = db_path
        self._default_ttl = default_ttl
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    model TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    ttl REAL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def put(
        self,
        messages: list[dict[str, Any]],
        response: Any,
        model: str = "",
        ttl: Optional[float] = None,
        **extras: Any,
    ) -> str:
        key = _hash_request(messages, model=model, **extras)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache (key, response, model, created_at, ttl, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (key, json.dumps(response), model, time.time(), effective_ttl))
            conn.commit()
        return key

    def get(self, messages: list[dict[str, Any]], model: str = "", **extras: Any) -> Optional[Any]:
        key = _hash_request(messages, model=model, **extras)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT response, created_at, ttl FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            response_json, created_at, ttl = row
            if ttl is not None and (time.time() - created_at) >= ttl:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            conn.execute("UPDATE cache SET hit_count = hit_count + 1 WHERE key = ?", (key,))
            conn.commit()
        return json.loads(response_json)

    def has(self, messages: list[dict[str, Any]], model: str = "", **extras: Any) -> bool:
        return self.get(messages, model=model, **extras) is not None

    def delete(self, messages: list[dict[str, Any]], model: str = "", **extras: Any) -> bool:
        key = _hash_request(messages, model=model, **extras)
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    def prune_expired(self) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM cache WHERE ttl IS NOT NULL AND (? - created_at) >= ttl",
                (time.time(),)
            )
            conn.commit()
            return cur.rowcount

    def clear(self) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM cache")
            conn.commit()
            return cur.rowcount

    @property
    def size(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return row[0] if row else 0

    def stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*), SUM(hit_count) FROM cache").fetchone()
            count, total_hits = row if row else (0, 0)
        return {
            "size": count,
            "total_hits": int(total_hits or 0),
            "db_path": self._db_path,
        }


__all__ = ["DiskResponseCache", "CacheEntry"]
