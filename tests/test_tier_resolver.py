"""Tests for TierResolver — the shared api_key → tier cache."""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from clsplusplus.tier_resolver import TierResolver


class FakeStore:
    def __init__(
        self,
        mapping: dict[str, Optional[str]],
        raises_for: Optional[str] = None,
        owners: Optional[dict[str, Optional[str]]] = None,
    ):
        self.mapping = mapping
        self.owners = owners or {}
        self.raises_for = raises_for
        self.calls = 0
        self.owner_calls = 0

    async def resolve_tier_from_key(self, raw_key: str) -> Optional[str]:
        self.calls += 1
        if self.raises_for and raw_key == self.raises_for:
            raise RuntimeError("db down")
        return self.mapping.get(raw_key)

    async def resolve_owner_email_from_key(self, raw_key: str) -> Optional[str]:
        self.owner_calls += 1
        if self.raises_for and raw_key == self.raises_for:
            raise RuntimeError("db down")
        return self.owners.get(raw_key)


@pytest.mark.asyncio
async def test_resolve_empty_key_returns_none():
    r = TierResolver(FakeStore({}))
    assert await r.resolve("") is None
    assert await r.resolve(None) is None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_resolve_hits_store_and_caches():
    store = FakeStore({"k1": "pro"})
    r = TierResolver(store, cache_ttl_seconds=60)
    assert await r.resolve("k1") == "pro"
    assert await r.resolve("k1") == "pro"
    assert store.calls == 1


@pytest.mark.asyncio
async def test_resolve_misses_are_also_cached():
    """`None` from the store means "no such user" — don't re-query for this."""
    store = FakeStore({"known": "pro"})
    r = TierResolver(store, cache_ttl_seconds=60)
    assert await r.resolve("unknown") is None
    assert await r.resolve("unknown") is None
    assert store.calls == 1


@pytest.mark.asyncio
async def test_invalidate_all_drops_every_entry():
    store = FakeStore({"k1": "pro", "k2": "business"})
    r = TierResolver(store, cache_ttl_seconds=60)
    await r.resolve("k1")
    await r.resolve("k2")
    assert store.calls == 2
    r.invalidate()
    await r.resolve("k1")
    await r.resolve("k2")
    assert store.calls == 4


@pytest.mark.asyncio
async def test_invalidate_scoped_to_one_key():
    store = FakeStore({"k1": "pro", "k2": "business"})
    r = TierResolver(store, cache_ttl_seconds=60)
    await r.resolve("k1")
    await r.resolve("k2")
    assert store.calls == 2
    r.invalidate("k1")
    await r.resolve("k1")  # re-queries
    await r.resolve("k2")  # still cached
    assert store.calls == 3


@pytest.mark.asyncio
async def test_store_error_returns_none_and_does_not_raise():
    store = FakeStore({}, raises_for="k1")
    r = TierResolver(store, cache_ttl_seconds=60)
    # Exception bubbles up from resolve? No — the resolver swallows it.
    assert await r.resolve("k1") is None


@pytest.mark.asyncio
async def test_cache_entry_expires_and_re_queries():
    """Entries age out after the TTL window."""
    store = FakeStore({"k1": "pro"})
    r = TierResolver(store, cache_ttl_seconds=0)  # immediate expiry
    await r.resolve("k1")
    # Minor yield to make sure monotonic() advances past 0.
    await asyncio.sleep(0)
    await r.resolve("k1")
    assert store.calls == 2


# --------------------------------------------------------------------------- #
# Owner + billing subject resolution
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resolve_owner_hits_store_and_caches():
    store = FakeStore({}, owners={"k1": "alice@example.com"})
    r = TierResolver(store, cache_ttl_seconds=60)
    assert await r.resolve_owner("k1") == "alice@example.com"
    assert await r.resolve_owner("k1") == "alice@example.com"
    assert store.owner_calls == 1


@pytest.mark.asyncio
async def test_resolve_owner_legacy_key_returns_none():
    """A key with no integration owner resolves to None."""
    store = FakeStore({}, owners={})
    r = TierResolver(store, cache_ttl_seconds=60)
    assert await r.resolve_owner("legacy-key") is None


@pytest.mark.asyncio
async def test_resolve_owner_db_error_returns_none():
    store = FakeStore({}, raises_for="k1")
    r = TierResolver(store, cache_ttl_seconds=60)
    assert await r.resolve_owner("k1") is None


@pytest.mark.asyncio
async def test_resolve_subject_owner_keys_collapse_per_user():
    """Two keys owned by the same user resolve to the SAME subject."""
    store = FakeStore(
        {},
        owners={"k1": "alice@example.com", "k2": "alice@example.com"},
    )
    r = TierResolver(store, cache_ttl_seconds=60)
    s1 = await r.resolve_subject("k1")
    s2 = await r.resolve_subject("k2")
    assert s1 == s2
    assert s1.startswith("owner:")


@pytest.mark.asyncio
async def test_resolve_subject_legacy_key_is_per_key():
    """Legacy keys with no owner get an isolated per-key subject."""
    store = FakeStore({}, owners={})
    r = TierResolver(store, cache_ttl_seconds=60)
    s1 = await r.resolve_subject("legacy-1")
    s2 = await r.resolve_subject("legacy-2")
    assert s1 != s2
    assert s1.startswith("key:")
    assert s2.startswith("key:")


@pytest.mark.asyncio
async def test_invalidate_drops_owner_cache_too():
    store = FakeStore({}, owners={"k1": "alice@example.com"})
    r = TierResolver(store, cache_ttl_seconds=60)
    await r.resolve_owner("k1")
    assert store.owner_calls == 1
    r.invalidate()
    await r.resolve_owner("k1")
    assert store.owner_calls == 2
