"""Tests for llm_response_cache_disk.DiskResponseCache."""

import threading
import time

import pytest

from llm_response_cache_disk import CacheStats, DiskResponseCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache(tmp_path):
    """Fresh cache backed by a temp SQLite file."""
    return DiskResponseCache(str(tmp_path / "cache.db"))


# ---------------------------------------------------------------------------
# Basic set / get
# ---------------------------------------------------------------------------


def test_set_get_round_trip(cache):
    cache.set("k1", "hello")
    assert cache.get("k1") == "hello"


def test_get_miss_returns_none(cache):
    assert cache.get("no-such-key") is None


def test_get_expired_returns_none(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=1)
    c.set("expiring", "value")
    time.sleep(1.05)
    assert c.get("expiring") is None


def test_expired_entry_deleted_from_db(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=1)
    c.set("temp", "gone")
    time.sleep(1.05)
    c.get("temp")  # triggers deletion
    assert c.size() == 0


def test_non_expired_entry_survives(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=60)
    c.set("alive", "yes")
    time.sleep(0.05)
    assert c.get("alive") == "yes"


def test_overwrite_existing_key(cache):
    cache.set("k", "first")
    cache.set("k", "second")
    assert cache.get("k") == "second"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_existing_returns_true(cache):
    cache.set("k", "v")
    assert cache.delete("k") is True


def test_delete_existing_removes_entry(cache):
    cache.set("k", "v")
    cache.delete("k")
    assert cache.get("k") is None


def test_delete_missing_returns_false(cache):
    assert cache.delete("nope") is False


# ---------------------------------------------------------------------------
# clear / size
# ---------------------------------------------------------------------------


def test_clear_removes_all(cache):
    cache.set("a", "1")
    cache.set("b", "2")
    cache.clear()
    assert cache.size() == 0


def test_size_correct_count(cache):
    cache.set("x", "1")
    cache.set("y", "2")
    cache.set("z", "3")
    assert cache.size() == 3


def test_size_includes_expired(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=1)
    c.set("exp", "v")
    time.sleep(1.05)
    # Expired but not yet cleaned up — size() counts raw DB rows.
    assert c.size() == 1


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_initial_zero(cache):
    s = cache.stats()
    assert s.hits == 0
    assert s.misses == 0
    assert s.expired == 0


def test_stats_hit_counted(cache):
    cache.set("k", "v")
    cache.get("k")
    assert cache.stats().hits == 1


def test_stats_miss_counted(cache):
    cache.get("missing")
    assert cache.stats().misses == 1


def test_stats_expired_counted(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=1)
    c.set("e", "v")
    time.sleep(1.05)
    c.get("e")
    s = c.stats()
    assert s.expired == 1
    assert s.misses == 1


def test_stats_total_entries_from_db(cache):
    cache.set("a", "1")
    cache.set("b", "2")
    assert cache.stats().total_entries == 2


def test_stats_returns_cache_stats_instance(cache):
    assert isinstance(cache.stats(), CacheStats)


# ---------------------------------------------------------------------------
# cached() decorator
# ---------------------------------------------------------------------------


def test_cached_miss_calls_fn(cache):
    call_count = {"n": 0}

    @cache.cached()
    def expensive(x: int) -> str:
        call_count["n"] += 1
        return f"result-{x}"

    result = expensive(42)
    assert result == "result-42"
    assert call_count["n"] == 1


def test_cached_hit_skips_fn(cache):
    call_count = {"n": 0}

    @cache.cached()
    def expensive(x: int) -> str:
        call_count["n"] += 1
        return f"result-{x}"

    expensive(7)
    expensive(7)
    assert call_count["n"] == 1


def test_cached_different_args_both_called(cache):
    call_count = {"n": 0}

    @cache.cached()
    def fn(x: int) -> str:
        call_count["n"] += 1
        return str(x)

    fn(1)
    fn(2)
    assert call_count["n"] == 2


def test_cached_custom_key_fn(cache):
    call_count = {"n": 0}

    @cache.cached(key_fn=lambda prompt: f"custom:{prompt}")
    def call_api(prompt: str) -> str:
        call_count["n"] += 1
        return f"answer-to-{prompt}"

    call_api("hello")
    call_api("hello")
    assert call_count["n"] == 1
    assert "custom:hello" in cache


# ---------------------------------------------------------------------------
# max_entries eviction
# ---------------------------------------------------------------------------


def test_max_entries_evicts_oldest(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), max_entries=3, ttl_seconds=3600)
    for i in range(4):
        c.set(f"key{i}", f"val{i}")
        time.sleep(0.01)  # ensure distinct expires_at ordering
    # After inserting 4 entries with max=3, one should be evicted.
    assert c.size() == 3


def test_max_entries_exactly_at_limit_no_eviction(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), max_entries=3, ttl_seconds=3600)
    for i in range(3):
        c.set(f"k{i}", f"v{i}")
    assert c.size() == 3


# ---------------------------------------------------------------------------
# invalidate_expired
# ---------------------------------------------------------------------------


def test_invalidate_expired_returns_count(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=1)
    c.set("a", "1")
    c.set("b", "2")
    time.sleep(1.05)
    removed = c.invalidate_expired()
    assert removed == 2


def test_invalidate_expired_removes_entries(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=1)
    c.set("x", "val")
    time.sleep(1.05)
    c.invalidate_expired()
    assert c.size() == 0


def test_invalidate_expired_keeps_valid(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=3600)
    c.set("good", "keep me")
    c.invalidate_expired()
    assert c.get("good") == "keep me"


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------


def test_contains_valid_key(cache):
    cache.set("present", "v")
    assert "present" in cache


def test_contains_missing_key(cache):
    assert "absent" not in cache


def test_contains_expired_key(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=1)
    c.set("gone", "v")
    time.sleep(1.05)
    assert "gone" not in c


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_thread_safety_concurrent_sets(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"))
    errors = []

    def worker(tid: int) -> None:
        try:
            for i in range(50):
                c.set(f"t{tid}-k{i}", f"value-{tid}-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"
    assert c.size() == 500


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------


def test_persist_across_instances(tmp_path):
    db = str(tmp_path / "persistent.db")
    c1 = DiskResponseCache(db)
    c1.set("persist-key", "persist-value")

    c2 = DiskResponseCache(db)
    assert c2.get("persist-key") == "persist-value"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_tilde_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    c = DiskResponseCache("~/tilde_test/cache.db")
    c.set("k", "v")
    assert c.get("k") == "v"


def test_parent_dirs_auto_created(tmp_path):
    deep_path = str(tmp_path / "a" / "b" / "c" / "cache.db")
    c = DiskResponseCache(deep_path)
    c.set("k", "v")
    assert c.get("k") == "v"


# ---------------------------------------------------------------------------
# Per-entry TTL override
# ---------------------------------------------------------------------------


def test_per_entry_ttl_override(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=3600)
    c.set("short", "value", ttl_seconds=1)
    time.sleep(1.05)
    # Short-lived entry should be gone; default TTL would still be valid.
    assert c.get("short") is None


def test_per_entry_ttl_longer_than_default(tmp_path):
    c = DiskResponseCache(str(tmp_path / "cache.db"), ttl_seconds=1)
    c.set("long", "value", ttl_seconds=3600)
    time.sleep(1.05)
    # Should still be valid despite default TTL expiring.
    assert c.get("long") == "value"
