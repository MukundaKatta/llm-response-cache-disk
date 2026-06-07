"""llm-response-cache-disk: SQLite-backed disk cache for LLM responses.

A tiny, dependency-free cache that persists LLM responses to a local SQLite
database. Responses are keyed by a stable SHA-256 hash of the request
(messages + model + any extra parameters such as ``temperature``), so identical
requests return the stored response instead of hitting the provider again.

Example::

    from llm_response_cache_disk import DiskResponseCache

    cache = DiskResponseCache("cache.db", default_ttl=3600)

    messages = [{"role": "user", "content": "Hello"}]
    if (response := cache.get(messages, model="claude")) is None:
        response = call_your_llm(messages)          # your own call
        cache.put(messages, response, model="claude")
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Iterator, Optional

__version__ = "0.1.0"


def _hash_request(messages: list[dict[str, Any]], model: str = "", **extras: Any) -> str:
    """Return a stable SHA-256 hex digest for a request.

    The digest is independent of dictionary ordering (``sort_keys=True``) so
    semantically identical requests always map to the same cache key. Any
    keyword in ``extras`` (for example ``temperature`` or ``max_tokens``)
    becomes part of the key, allowing the same prompt under different sampling
    parameters to be cached separately.
    """
    payload = {"messages": messages, "model": model, **extras}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass
class CacheEntry:
    """A single stored cache record, as returned by :meth:`DiskResponseCache.get_entry`."""

    key: str
    response: Any
    created_at: float
    ttl: Optional[float]
    hit_count: int = 0
    model: str = ""

    @property
    def expired(self) -> bool:
        """Whether this entry has passed its TTL. Entries with no TTL never expire."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) >= self.ttl


class DiskResponseCache:
    """SQLite-backed disk cache for LLM responses. Persists across restarts.

    Args:
        db_path: Filesystem path to the SQLite database file. Parent
            directories are created automatically if they do not exist.
        default_ttl: Default time-to-live in seconds applied to entries that do
            not specify their own ``ttl``. ``None`` (the default) means entries
            never expire.
        timeout: Seconds SQLite waits for a lock before raising
            ``sqlite3.OperationalError``. Defaults to 30 seconds, which makes
            the cache tolerant of concurrent readers/writers.

    Example::

        cache = DiskResponseCache("/tmp/llm-cache.db", default_ttl=3600)
        key = cache.put(messages, response, model="claude")
        cached = cache.get(messages, model="claude")

    The instance can also be used as a context manager::

        with DiskResponseCache("cache.db") as cache:
            cache.put(messages, response, model="claude")
    """

    def __init__(
        self,
        db_path: str,
        default_ttl: Optional[float] = None,
        timeout: float = 30.0,
    ) -> None:
        self._db_path = db_path
        self._default_ttl = default_ttl
        self._timeout = timeout
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=self._timeout)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    model TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    ttl REAL,
                    hit_count INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()

    def put(
        self,
        messages: list[dict[str, Any]],
        response: Any,
        model: str = "",
        ttl: Optional[float] = None,
        **extras: Any,
    ) -> str:
        """Store ``response`` for the given request and return its cache key.

        Args:
            messages: The request messages used to compute the cache key.
            response: Any JSON-serializable value to cache.
            model: Model identifier; included in the cache key so the same
                prompt cached under different models does not collide.
            ttl: Per-entry time-to-live in seconds. Falls back to
                ``default_ttl`` when ``None``.
            **extras: Extra request parameters (e.g. ``temperature``) folded
                into the cache key.

        Returns:
            The SHA-256 cache key under which the response was stored.

        Raises:
            TypeError: If ``response`` or any part of the request is not
                JSON-serializable.
        """
        key = _hash_request(messages, model=model, **extras)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, response, model, created_at, ttl, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (key, json.dumps(response), model, time.time(), effective_ttl),
            )
            conn.commit()
        return key

    def get(
        self, messages: list[dict[str, Any]], model: str = "", **extras: Any
    ) -> Optional[Any]:
        """Return the cached response for a request, or ``None`` on a miss.

        Expired entries are treated as a miss and removed lazily. A successful
        lookup increments the entry's ``hit_count``.
        """
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
            conn.execute(
                "UPDATE cache SET hit_count = hit_count + 1 WHERE key = ?", (key,)
            )
            conn.commit()
        return json.loads(response_json)

    def get_entry(
        self, messages: list[dict[str, Any]], model: str = "", **extras: Any
    ) -> Optional[CacheEntry]:
        """Return the full :class:`CacheEntry` for a request, or ``None``.

        Unlike :meth:`get`, this does not increment ``hit_count``, does not
        prune expired rows, and exposes metadata (creation time, TTL, hit
        count). Use :attr:`CacheEntry.expired` to check freshness yourself.
        """
        key = _hash_request(messages, model=model, **extras)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT response, model, created_at, ttl, hit_count "
                "FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        response_json, entry_model, created_at, ttl, hit_count = row
        return CacheEntry(
            key=key,
            response=json.loads(response_json),
            created_at=created_at,
            ttl=ttl,
            hit_count=hit_count,
            model=entry_model,
        )

    def has(
        self, messages: list[dict[str, Any]], model: str = "", **extras: Any
    ) -> bool:
        """Return ``True`` if a non-expired response is cached for the request.

        Note that, like :meth:`get`, calling this on a hit increments the
        entry's ``hit_count`` and prunes the entry if it has expired.
        """
        return self.get(messages, model=model, **extras) is not None

    def delete(
        self, messages: list[dict[str, Any]], model: str = "", **extras: Any
    ) -> bool:
        """Delete the cached entry for a request. Returns ``True`` if one existed."""
        key = _hash_request(messages, model=model, **extras)
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    def prune_expired(self) -> int:
        """Delete all expired entries and return how many were removed."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM cache WHERE ttl IS NOT NULL AND (? - created_at) >= ttl",
                (time.time(),),
            )
            conn.commit()
            return cur.rowcount

    def clear(self) -> int:
        """Delete every entry and return how many were removed."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM cache")
            conn.commit()
            return cur.rowcount

    @property
    def size(self) -> int:
        """Total number of stored entries, including any that have expired."""
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return row[0] if row else 0

    def keys(self) -> list[str]:
        """Return the cache keys of all stored entries."""
        with self._conn() as conn:
            rows = conn.execute("SELECT key FROM cache").fetchall()
        return [r[0] for r in rows]

    def stats(self) -> dict[str, Any]:
        """Return summary statistics: entry count, total hits, and db path."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*), SUM(hit_count) FROM cache"
            ).fetchone()
            count, total_hits = row if row else (0, 0)
        return {
            "size": count,
            "total_hits": int(total_hits or 0),
            "db_path": self._db_path,
        }

    def close(self) -> None:
        """Release resources held by the cache.

        Provided for API symmetry and context-manager support. Connections are
        opened per operation and closed automatically, so this is a no-op, but
        callers may still invoke it (or use ``with``) for clarity.
        """
        return None

    def __len__(self) -> int:
        return self.size

    def __enter__(self) -> "DiskResponseCache":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __repr__(self) -> str:
        return (
            f"DiskResponseCache(db_path={self._db_path!r}, "
            f"default_ttl={self._default_ttl!r}, size={self.size})"
        )


__all__ = ["DiskResponseCache", "CacheEntry", "__version__"]
