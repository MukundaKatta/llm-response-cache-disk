"""Tests for llm-response-cache-disk.

These tests use only the Python standard library (``unittest`` + ``tempfile``)
so they run without any third-party dependencies::

    python3 -m unittest discover -s tests
"""
import os
import sys
import tempfile
import time
import unittest

# Make ``src`` importable when running the tests directly from a checkout
# (e.g. ``python3 -m unittest discover -s tests``) without installing the
# package first.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_response_cache_disk import (  # noqa: E402
    CacheEntry,
    DiskResponseCache,
    __version__,
)

MSGS = [{"role": "user", "content": "Hello"}]
RESPONSE = {"content": "Hi there!"}


class DiskResponseCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = os.path.join(self._tmp.name, "cache.db")

    def make(self, **kwargs) -> DiskResponseCache:
        return DiskResponseCache(self.db, **kwargs)

    # -- basic put / get -------------------------------------------------

    def test_put_and_get(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="claude")
        self.assertEqual(cache.get(MSGS, model="claude"), RESPONSE)

    def test_put_returns_stable_key(self):
        cache = self.make()
        k1 = cache.put(MSGS, RESPONSE, model="claude")
        k2 = cache.put(MSGS, {"content": "different"}, model="claude")
        self.assertEqual(k1, k2)  # same request -> same key (overwrite)
        self.assertEqual(cache.get(MSGS, model="claude"), {"content": "different"})

    def test_get_miss_returns_none(self):
        self.assertIsNone(self.make().get(MSGS, model="claude"))

    def test_get_is_order_independent(self):
        cache = self.make()
        msg = [{"role": "user", "content": "x"}]
        cache.put(msg, RESPONSE, model="m", temperature=0.5, top_p=0.9)
        # Same kwargs in a different order must hit the same key.
        self.assertEqual(cache.get(msg, model="m", top_p=0.9, temperature=0.5), RESPONSE)

    # -- has --------------------------------------------------------------

    def test_has_true(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="claude")
        self.assertTrue(cache.has(MSGS, model="claude"))

    def test_has_false(self):
        self.assertFalse(self.make().has(MSGS, model="claude"))

    # -- size / len / keys / iter ----------------------------------------

    def test_size(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="a")
        cache.put([{"role": "user", "content": "other"}], RESPONSE, model="b")
        self.assertEqual(cache.size, 2)
        self.assertEqual(len(cache), 2)

    def test_keys_and_iter(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="a")
        cache.put([{"role": "user", "content": "other"}], RESPONSE, model="b")
        keys = cache.keys()
        self.assertEqual(len(keys), 2)
        self.assertEqual(set(iter(cache)), set(keys))

    # -- delete / clear ---------------------------------------------------

    def test_delete(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="claude")
        self.assertTrue(cache.delete(MSGS, model="claude"))
        self.assertIsNone(cache.get(MSGS, model="claude"))

    def test_delete_missing(self):
        self.assertFalse(self.make().delete(MSGS, model="missing"))

    def test_clear(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="a")
        cache.put(MSGS, RESPONSE, model="b")
        self.assertEqual(cache.clear(), 2)
        self.assertEqual(cache.size, 0)

    # -- TTL --------------------------------------------------------------

    def test_default_ttl_expiry(self):
        cache = self.make(default_ttl=0.05)
        cache.put(MSGS, RESPONSE, model="claude")
        time.sleep(0.1)
        self.assertIsNone(cache.get(MSGS, model="claude"))

    def test_ttl_not_expired(self):
        cache = self.make(default_ttl=60.0)
        cache.put(MSGS, RESPONSE, model="claude")
        self.assertEqual(cache.get(MSGS, model="claude"), RESPONSE)

    def test_per_entry_ttl_overrides_default(self):
        cache = self.make(default_ttl=60.0)
        cache.put(MSGS, RESPONSE, model="claude", ttl=0.05)
        time.sleep(0.1)
        self.assertIsNone(cache.get(MSGS, model="claude"))

    def test_expired_entry_is_pruned_on_get(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="claude", ttl=0.05)
        time.sleep(0.1)
        self.assertIsNone(cache.get(MSGS, model="claude"))
        self.assertEqual(cache.size, 0)  # lazily removed

    def test_prune_expired(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="a", ttl=0.05)
        cache.put([{"role": "user", "content": "keep"}], RESPONSE, model="b", ttl=60.0)
        time.sleep(0.1)
        self.assertEqual(cache.prune_expired(), 1)
        self.assertEqual(cache.size, 1)

    # -- persistence ------------------------------------------------------

    def test_persists_across_instances(self):
        c1 = self.make()
        c1.put(MSGS, RESPONSE, model="claude")
        c2 = self.make()
        self.assertEqual(c2.get(MSGS, model="claude"), RESPONSE)

    def test_creates_nested_parent_dirs(self):
        nested = os.path.join(self._tmp.name, "a", "b", "c", "cache.db")
        cache = DiskResponseCache(nested)
        cache.put(MSGS, RESPONSE, model="m")
        self.assertTrue(os.path.exists(nested))
        self.assertEqual(cache.get(MSGS, model="m"), RESPONSE)

    # -- key differentiation ---------------------------------------------

    def test_model_differentiates_keys(self):
        cache = self.make()
        cache.put(MSGS, {"a": 1}, model="claude")
        cache.put(MSGS, {"b": 2}, model="gpt4")
        self.assertEqual(cache.get(MSGS, model="claude"), {"a": 1})
        self.assertEqual(cache.get(MSGS, model="gpt4"), {"b": 2})

    def test_extras_differentiate_keys(self):
        cache = self.make()
        cache.put(MSGS, {"hot": True}, model="m", temperature=1.0)
        cache.put(MSGS, {"hot": False}, model="m", temperature=0.0)
        self.assertEqual(cache.get(MSGS, model="m", temperature=1.0), {"hot": True})
        self.assertEqual(cache.get(MSGS, model="m", temperature=0.0), {"hot": False})

    # -- stats / hit counting --------------------------------------------

    def test_stats_empty(self):
        s = self.make().stats()
        self.assertEqual(s["size"], 0)
        self.assertEqual(s["total_hits"], 0)
        self.assertEqual(s["db_path"], self.db)

    def test_stats_counts_hits(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="x")
        cache.get(MSGS, model="x")
        cache.get(MSGS, model="x")
        s = cache.stats()
        self.assertEqual(s["size"], 1)
        self.assertEqual(s["total_hits"], 2)

    # -- get_entry --------------------------------------------------------

    def test_get_entry_returns_metadata(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="claude", ttl=60.0)
        entry = cache.get_entry(MSGS, model="claude")
        self.assertIsInstance(entry, CacheEntry)
        self.assertEqual(entry.response, RESPONSE)
        self.assertEqual(entry.model, "claude")
        self.assertEqual(entry.ttl, 60.0)
        self.assertFalse(entry.expired)

    def test_get_entry_miss(self):
        self.assertIsNone(self.make().get_entry(MSGS, model="nope"))

    def test_get_entry_does_not_increment_hits(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="x")
        cache.get_entry(MSGS, model="x")
        self.assertEqual(cache.stats()["total_hits"], 0)

    def test_get_entry_reports_expired(self):
        cache = self.make()
        cache.put(MSGS, RESPONSE, model="x", ttl=0.05)
        time.sleep(0.1)
        entry = cache.get_entry(MSGS, model="x")
        self.assertIsNotNone(entry)  # get_entry does NOT prune
        self.assertTrue(entry.expired)

    # -- context manager / misc ------------------------------------------

    def test_context_manager(self):
        with DiskResponseCache(self.db) as cache:
            cache.put(MSGS, RESPONSE, model="m")
            self.assertEqual(cache.get(MSGS, model="m"), RESPONSE)

    def test_repr_contains_path(self):
        cache = self.make()
        self.assertIn("DiskResponseCache", repr(cache))
        self.assertIn(self.db, repr(cache))

    def test_non_serializable_response_raises(self):
        cache = self.make()
        with self.assertRaises(TypeError):
            cache.put(MSGS, {1, 2, 3}, model="m")  # set is not JSON-serializable


class CacheEntryTestCase(unittest.TestCase):
    def test_no_ttl_never_expires(self):
        entry = CacheEntry(key="k", response={}, created_at=0.0, ttl=None)
        self.assertFalse(entry.expired)

    def test_ttl_expired(self):
        entry = CacheEntry(key="k", response={}, created_at=time.time() - 100, ttl=1.0)
        self.assertTrue(entry.expired)

    def test_ttl_not_expired(self):
        entry = CacheEntry(key="k", response={}, created_at=time.time(), ttl=1000.0)
        self.assertFalse(entry.expired)


class MetadataTestCase(unittest.TestCase):
    def test_version_is_string(self):
        self.assertIsInstance(__version__, str)


if __name__ == "__main__":
    unittest.main()
