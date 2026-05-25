# llm-response-cache-disk

[![PyPI](https://img.shields.io/pypi/v/llm-response-cache-disk.svg)](https://pypi.org/project/llm-response-cache-disk/)
[![Python](https://img.shields.io/pypi/pyversions/llm-response-cache-disk.svg)](https://pypi.org/project/llm-response-cache-disk/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SQLite-backed disk cache for LLM text responses.**

Complements in-process LRU caches by persisting responses across process
restarts. Useful during development to avoid repeated API calls while
iterating on prompts, agents, or pipelines.

Zero runtime dependencies (uses stdlib `sqlite3`). Thread-safe. Python 3.10+.

## Install

```bash
pip install llm-response-cache-disk
```

## Quick start

```python
from llm_response_cache_disk import DiskResponseCache

cache = DiskResponseCache("~/.cache/llm/responses.db", ttl_seconds=86_400)

# Basic get/set
cache.set("my-key", "some LLM response")
value = cache.get("my-key")   # returns None on miss or expiry

# Decorator — wraps any sync str-returning function
@cache.cached()
def call_api(prompt: str) -> str:
    return my_llm_client.complete(prompt)

response = call_api("Tell me about caching")  # first call hits the API
response = call_api("Tell me about caching")  # second call returns from disk

# Custom cache key
import hashlib

@cache.cached(key_fn=lambda prompt: hashlib.sha256(prompt.encode()).hexdigest())
def call_with_custom_key(prompt: str) -> str:
    return my_llm_client.complete(prompt)
```

## API

### `DiskResponseCache(path, max_entries=10_000, ttl_seconds=86_400)`

- `path` — path to the SQLite file; `~` is expanded, parent directories are created automatically.
- `max_entries` — maximum entries to keep; oldest (by expiry) are evicted when exceeded.
- `ttl_seconds` — default time-to-live per entry.

| Method | Description |
|---|---|
| `get(key)` | Return cached value or `None` on miss/expiry. Expired entries are deleted on access. |
| `set(key, value, ttl_seconds=None)` | Store value. `ttl_seconds` overrides the instance default for this entry. |
| `delete(key)` | Remove entry. Returns `True` if it existed. |
| `clear()` | Remove all entries. |
| `size()` | Total entries in DB (including expired). |
| `stats()` | Returns `CacheStats(hits, misses, expired, total_entries)`. |
| `invalidate_expired()` | Delete all expired entries. Returns count removed. |
| `cached(key_fn=None)` | Decorator for sync str-returning functions. |
| `key in cache` | `True` if key exists and has not expired. |

## License

MIT
