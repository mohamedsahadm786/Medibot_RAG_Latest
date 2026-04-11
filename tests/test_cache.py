"""
Tests for backend/services/cache.py (Phase 9).
All Redis and embedding calls are mocked — no real Redis or model usage.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.services.cache import (
    CACHE_THRESHOLD,
    MAX_CACHE_ENTRIES,
    _INDEX_KEY,
    check_cache,
    clear_cache,
    store_cache,
)

# ── Fixed unit vectors for deterministic similarity tests ──────────────────────
# Both normalised, so cosine similarity == dot product.

_VEC_A = np.array([1.0, 0.0, 0.0], dtype=np.float32)          # query vector
_VEC_SAME = np.array([1.0, 0.0, 0.0], dtype=np.float32)        # identical → sim 1.0
_VEC_CLOSE = np.array([0.98, 0.2, 0.0], dtype=np.float32)      # similar   → sim 0.98
_VEC_ORTHO = np.array([0.0, 1.0, 0.0], dtype=np.float32)       # orthogonal → sim 0.0

# Normalise _VEC_CLOSE so dot product equals cosine similarity
_VEC_CLOSE = (_VEC_CLOSE / np.linalg.norm(_VEC_CLOSE)).astype(np.float32)


@pytest.fixture
def redis_mock():
    """Provide a fully-mocked async Redis client and patch _get_redis."""
    mock = AsyncMock()
    mock.zrange = AsyncMock(return_value=[])
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.zadd = AsyncMock()
    mock.zcard = AsyncMock(return_value=0)
    mock.zremrangebyrank = AsyncMock()
    mock.delete = AsyncMock()
    with patch("backend.services.cache._get_redis", new=AsyncMock(return_value=mock)):
        yield mock


# ── check_cache ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_cache_returns_none_when_index_empty(redis_mock) -> None:
    """Cache miss when no entries exist in the index."""
    redis_mock.zrange.return_value = []
    with patch("backend.services.cache._embed", return_value=_VEC_A):
        result = await check_cache("What is diabetes?")
    assert result is None


@pytest.mark.asyncio
async def test_check_cache_returns_hit_on_identical_query(redis_mock) -> None:
    """Exact match (similarity=1.0) should return the cached answer dict."""
    cached_payload = {"answer": "Diabetes is a metabolic disease.", "sources": []}
    redis_mock.zrange.return_value = [b"entry_abc"]
    redis_mock.get = AsyncMock(side_effect=[
        _VEC_SAME.tobytes(),                      # first get → embedding bytes
        json.dumps(cached_payload).encode(),       # second get → answer JSON
    ])
    with patch("backend.services.cache._embed", return_value=_VEC_A):
        result = await check_cache("What is diabetes?")
    assert result is not None
    assert result["answer"] == cached_payload["answer"]


@pytest.mark.asyncio
async def test_check_cache_returns_hit_above_threshold(redis_mock) -> None:
    """Similarity above threshold (but not 1.0) should still be a cache hit."""
    cached_payload = {"answer": "Some cached answer.", "sources": []}
    redis_mock.zrange.return_value = [b"entry_xyz"]
    redis_mock.get = AsyncMock(side_effect=[
        _VEC_CLOSE.tobytes(),
        json.dumps(cached_payload).encode(),
    ])
    with patch("backend.services.cache._embed", return_value=_VEC_A):
        result = await check_cache("Similar query")
    # _VEC_A · _VEC_CLOSE ≈ 0.98 which is > CACHE_THRESHOLD (0.95)
    assert result is not None


@pytest.mark.asyncio
async def test_check_cache_returns_none_on_low_similarity(redis_mock) -> None:
    """Orthogonal query (similarity=0.0) should miss the cache."""
    redis_mock.zrange.return_value = [b"entry_1"]
    redis_mock.get = AsyncMock(return_value=_VEC_ORTHO.tobytes())
    with patch("backend.services.cache._embed", return_value=_VEC_A):
        result = await check_cache("Completely different topic")
    assert result is None


@pytest.mark.asyncio
async def test_check_cache_skips_expired_embedding(redis_mock) -> None:
    """Entry ID present in index but embedding key expired → graceful miss."""
    redis_mock.zrange.return_value = [b"expired_entry"]
    redis_mock.get.return_value = None  # key expired
    with patch("backend.services.cache._embed", return_value=_VEC_A):
        result = await check_cache("Any query")
    assert result is None


# ── store_cache ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_cache_writes_two_keys_with_ttl(redis_mock) -> None:
    """store_cache must call redis.set exactly twice, both with 24-hour TTL."""
    redis_mock.zcard.return_value = 1  # well under limit
    with patch("backend.services.cache._embed", return_value=_VEC_A):
        await store_cache("What is diabetes?", "It is a chronic disease.", [])

    assert redis_mock.set.call_count == 2
    for call in redis_mock.set.call_args_list:
        assert call.kwargs.get("ex") == 86400


@pytest.mark.asyncio
async def test_store_cache_evicts_oldest_when_full(redis_mock) -> None:
    """When entry count exceeds MAX_CACHE_ENTRIES, the oldest entry is removed."""
    redis_mock.zcard.return_value = MAX_CACHE_ENTRIES + 1
    redis_mock.zrange.return_value = [b"oldest_entry"]

    with patch("backend.services.cache._embed", return_value=_VEC_A):
        await store_cache("New query", "New answer", [])

    # The oldest entry's two keys should be deleted together
    redis_mock.delete.assert_called_once_with(
        "cache:embedding:oldest_entry",
        "cache:answer:oldest_entry",
    )
    # And the sorted set should be trimmed
    redis_mock.zremrangebyrank.assert_called_once_with(_INDEX_KEY, 0, 0)


# ── clear_cache ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_cache_removes_all_entries(redis_mock) -> None:
    """clear_cache must delete both data keys per entry and the index key."""
    redis_mock.zrange.return_value = [b"entry1", b"entry2"]

    count = await clear_cache()

    assert count == 2
    # 2 per-entry delete calls + 1 index key delete
    assert redis_mock.delete.call_count == 3
    redis_mock.delete.assert_any_call("cache:embedding:entry1", "cache:answer:entry1")
    redis_mock.delete.assert_any_call("cache:embedding:entry2", "cache:answer:entry2")
    redis_mock.delete.assert_any_call(_INDEX_KEY)


@pytest.mark.asyncio
async def test_clear_cache_returns_zero_when_empty(redis_mock) -> None:
    """clear_cache on an empty cache returns 0."""
    redis_mock.zrange.return_value = []

    count = await clear_cache()

    assert count == 0
    # Only the index key delete is called (on an empty set it's a no-op)
    redis_mock.delete.assert_called_once_with(_INDEX_KEY)
