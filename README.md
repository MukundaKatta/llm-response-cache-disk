# llm-response-cache-disk

[![CI](https://github.com/MukundaKatta/llm-response-cache-disk/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/llm-response-cache-disk/actions/workflows/ci.yml)

SQLite-backed disk cache for LLM responses. **Zero dependencies. Python 3.10+.**

LLM calls are slow and expensive, and during development you often send the
*exact same request* over and over. `llm-response-cache-disk` stores each
response in a local SQLite file keyed by a stable hash of the request
(messages + model + any extra parameters such as `temperature`). Identical
requests return the cached response instantly, and because the cache lives on
disk it survives process restarts.

- **No third-party dependencies** — only the Python standard library (`sqlite3`, `hashlib`, `json`).
- **Persistent** — backed by a single SQLite file; works across runs and processes.
- **Deterministic keys** — order-independent hashing so semantically identical requests collide on purpose.
- **Optional TTLs** — global default and per-entry expiry, with lazy and bulk pruning.
- **Provider-agnostic** — caches any JSON-serializable response from any LLM SDK.

## Installation

Install from a clone of this repository:

```bash
git clone https://github.com/MukundaKatta/llm-response-cache-disk.git
cd llm-response-cache-disk
pip install .
```

Or install the latest from GitHub directly:

```bash
pip install "git+https://github.com/MukundaKatta/llm-response-cache-disk.git"
```

There are no runtime dependencies to install.

## Quick start

```python
from llm_response_cache_disk import DiskResponseCache

# Entries expire after one hour by default; omit default_ttl to keep forever.
cache = DiskResponseCache("cache.db", default_ttl=3600)

messages = [{"role": "user", "content": "Explain caching in one sentence."}]

# Look up before calling your LLM; only call on a miss.
response = cache.get(messages, model="claude")
if response is None:
    response = call_your_llm(messages)          # your own provider call
    cache.put(messages, response, model="claude")

print(response)
```

### Wrapping any LLM client

The cache does not know or care which provider you use — it just stores the
JSON-serializable object you give it:

```python
from llm_response_cache_disk import DiskResponseCache

cache = DiskResponseCache("cache.db")

def cached_chat(client, messages, model, **params):
    """Return a cached completion if available, otherwise call and store it."""
    hit = cache.get(messages, model=model, **params)
    if hit is not None:
        return hit
    result = client.chat(messages=messages, model=model, **params)  # your SDK
    cache.put(messages, result, model=model, **params)
    return result
```

Any extra keyword arguments (for example `temperature` or `max_tokens`) become
part of the cache key, so the same prompt under different sampling settings is
cached separately.

### Use it as a context manager

```python
with DiskResponseCache("cache.db") as cache:
    cache.put(messages, response, model="claude")
```

## How keys work

A cache key is the SHA-256 hex digest of a canonical JSON encoding of
`{"messages": ..., "model": ..., **extras}`. Encoding uses `sort_keys=True`, so
dictionary ordering never affects the key — two requests that differ only in key
order map to the same entry. Changing the model, the messages, or any extra
parameter produces a different key.

## API reference

### `DiskResponseCache(db_path, default_ttl=None, timeout=30.0)`

Create (or open) a cache backed by the SQLite file at `db_path`. Parent
directories are created automatically. `default_ttl` is the fallback
time-to-live in seconds for entries that don't set their own (`None` means
never expire). `timeout` is how long SQLite waits for a lock before raising.

| Method | Description |
| --- | --- |
| `put(messages, response, model="", ttl=None, **extras) -> str` | Store `response` and return its cache key. `ttl` overrides `default_ttl` for this entry. Raises `TypeError` if `response` is not JSON-serializable. |
| `get(messages, model="", **extras) -> Any \| None` | Return the cached response or `None`. Expired entries are pruned lazily and counted as a miss. A hit increments `hit_count`. |
| `get_entry(messages, model="", **extras) -> CacheEntry \| None` | Return the full record (with metadata) without incrementing hits or pruning expired entries. |
| `has(messages, model="", **extras) -> bool` | `True` if a non-expired entry exists. Like `get`, this counts as a hit and prunes expired entries. |
| `delete(messages, model="", **extras) -> bool` | Delete one entry; `True` if it existed. |
| `prune_expired() -> int` | Delete every expired entry; returns how many were removed. |
| `clear() -> int` | Delete all entries; returns how many were removed. |
| `keys() -> list[str]` | Return the cache keys of all stored entries. |
| `stats() -> dict` | `{"size", "total_hits", "db_path"}` summary. |
| `close() -> None` | Release resources (no-op; connections are per-operation). |

Special methods: `len(cache)` returns the number of entries, `iter(cache)`
yields cache keys, and the instance supports the `with` statement.

### `CacheEntry`

A dataclass returned by `get_entry`, with fields `key`, `response`,
`created_at`, `ttl`, `hit_count`, `model`, and an `expired` property that
reports whether the entry has passed its TTL.

## Running the tests

The test suite uses only the standard library, so no extra installs are needed:

```bash
python3 -m unittest discover -s tests
```

## License

MIT — see [LICENSE](LICENSE).
