import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["DiskResponseCache", "CacheStats"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at);
"""


@dataclass
class CacheStats:
    hits: int
    misses: int
    expired: int
    total_entries: int


class DiskResponseCache:
    """SQLite-backed disk cache for LLM text responses.

    Responses persist across process restarts. Thread-safe via RLock.
    Zero runtime dependencies — uses stdlib sqlite3 only.

    Args:
        path: Path to the SQLite database file. ``~`` is expanded.
        max_entries: Maximum number of entries to keep. Oldest entries
            (by expires_at) are evicted when the limit is exceeded.
        ttl_seconds: Default time-to-live in seconds for each entry.
    """

    def __init__(
        self,
        path: str,
        max_entries: int = 10_000,
        ttl_seconds: int = 86_400,
    ) -> None:
        self._path = os.path.expanduser(path)
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

        # In-memory counters; reset when a new instance is created.
        self._hits = 0
        self._misses = 0
        self._expired = 0

        # Auto-create parent directories.
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Core get / set / delete
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Return cached value, or None on miss or expiry.

        Expired entries are deleted immediately and counted in stats.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()

            if row is None:
                self._misses += 1
                return None

            value, expires_at = row
            if time.time() > expires_at:
                # Delete the stale entry and report as expired.
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
                self._expired += 1
                self._misses += 1
                return None

            self._hits += 1
            return value

    def set(
        self,
        key: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store *value* under *key* with an optional per-entry TTL override."""
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
        now = time.time()
        expires_at = now + ttl

        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at, created_at)"
                " VALUES (?, ?, ?, ?)",
                (key, value, expires_at, now),
            )
            self._conn.commit()
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        """Remove *key* from the cache. Returns True if the key existed."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return the total number of entries in the DB (including expired)."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return row[0]

    def stats(self) -> CacheStats:
        """Return in-memory hit/miss/expired counters plus live DB entry count."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                expired=self._expired,
                total_entries=self.size(),
            )

    def invalidate_expired(self) -> int:
        """Delete all expired entries. Returns the number of rows removed."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM cache WHERE expires_at < ?", (time.time(),)
            )
            self._conn.commit()
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def cached(self, key_fn: Callable | None = None):  # noqa: B006
        """Decorator for sync string-returning functions.

        The cached result is stored using the default TTL. Use ``key_fn``
        to provide a custom cache-key function::

            @cache.cached(key_fn=lambda prompt: hashlib.sha256(prompt.encode()).hexdigest())
            def call_api(prompt: str) -> str:
                ...

        Default key: ``sha256("<fn_name>:" + json.dumps(sorted args+kwargs))``.
        """

        def decorator(fn: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                if key_fn is not None:
                    key = key_fn(*args, **kwargs)
                else:
                    payload = json.dumps(
                        {"args": list(args), "kwargs": sorted(kwargs.items())},
                        sort_keys=True,
                        default=str,
                    )
                    raw = f"{fn.__name__}:{payload}"
                    key = hashlib.sha256(raw.encode()).hexdigest()

                cached_value = self.get(key)
                if cached_value is not None:
                    return cached_value

                result: str = fn(*args, **kwargs)
                self.set(key, result)
                return result

            wrapper.__name__ = fn.__name__
            wrapper.__doc__ = fn.__doc__
            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Membership test
    # ------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        """Return True if *key* exists in the cache and has not expired."""
        with self._lock:
            row = self._conn.execute(
                "SELECT expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return False
            return time.time() <= row[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict oldest entries (by expires_at ASC) until size <= max_entries.

        Must be called while self._lock is already held.
        """
        count_row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        count = count_row[0]
        if count <= self._max_entries:
            return

        excess = count - self._max_entries
        # Delete the *excess* rows with the smallest expires_at (expire soonest).
        self._conn.execute(
            "DELETE FROM cache WHERE key IN ("
            "  SELECT key FROM cache ORDER BY expires_at ASC LIMIT ?"
            ")",
            (excess,),
        )
        self._conn.commit()
