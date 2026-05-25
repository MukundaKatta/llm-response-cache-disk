"""Tests for llm-response-cache-disk."""
import time
import pytest
from llm_response_cache_disk import DiskResponseCache, CacheEntry

MSGS = [{"role": "user", "content": "Hello"}]
RESPONSE = {"content": "Hi there!"}


def test_put_and_get(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    cache.put(MSGS, RESPONSE, model="claude")
    result = cache.get(MSGS, model="claude")
    assert result == RESPONSE


def test_get_miss(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    assert cache.get(MSGS, model="claude") is None


def test_has_true(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    cache.put(MSGS, RESPONSE, model="claude")
    assert cache.has(MSGS, model="claude") is True


def test_has_false(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    assert cache.has(MSGS, model="claude") is False


def test_size(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    cache.put(MSGS, RESPONSE, model="a")
    cache.put([{"role": "user", "content": "other"}], RESPONSE, model="b")
    assert cache.size == 2


def test_delete(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    cache.put(MSGS, RESPONSE, model="claude")
    assert cache.delete(MSGS, model="claude") is True
    assert cache.get(MSGS, model="claude") is None


def test_delete_missing(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    assert cache.delete(MSGS, model="missing") is False


def test_clear(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    cache.put(MSGS, RESPONSE, model="a")
    cache.put(MSGS, RESPONSE, model="b")
    count = cache.clear()
    assert count == 2
    assert cache.size == 0


def test_ttl_expiry(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db, default_ttl=0.05)
    cache.put(MSGS, RESPONSE, model="claude")
    time.sleep(0.1)
    assert cache.get(MSGS, model="claude") is None


def test_ttl_not_expired(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db, default_ttl=60.0)
    cache.put(MSGS, RESPONSE, model="claude")
    assert cache.get(MSGS, model="claude") == RESPONSE


def test_per_entry_ttl(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    cache.put(MSGS, RESPONSE, model="claude", ttl=0.05)
    time.sleep(0.1)
    assert cache.get(MSGS, model="claude") is None


def test_prune_expired(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    cache.put(MSGS, RESPONSE, model="a", ttl=0.05)
    cache.put([{"role": "user", "content": "keep"}], RESPONSE, model="b", ttl=60.0)
    time.sleep(0.1)
    removed = cache.prune_expired()
    assert removed == 1
    assert cache.size == 1


def test_persists_across_instances(tmp_path):
    db = str(tmp_path / "cache.db")
    c1 = DiskResponseCache(db)
    c1.put(MSGS, RESPONSE, model="claude")
    c2 = DiskResponseCache(db)
    assert c2.get(MSGS, model="claude") == RESPONSE


def test_model_differentiates_keys(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    cache.put(MSGS, {"a": 1}, model="claude")
    cache.put(MSGS, {"b": 2}, model="gpt4")
    assert cache.get(MSGS, model="claude") == {"a": 1}
    assert cache.get(MSGS, model="gpt4") == {"b": 2}


def test_stats(tmp_path):
    db = str(tmp_path / "cache.db")
    cache = DiskResponseCache(db)
    cache.put(MSGS, RESPONSE, model="x")
    cache.get(MSGS, model="x")
    s = cache.stats()
    assert s["size"] == 1
    assert s["total_hits"] >= 1
