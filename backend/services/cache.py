"""
MediBot v2 — Semantic Cache Service (Phase 9)

Redis-backed semantic cache with cosine similarity matching.

Before each RAG pipeline run, the raw query is embedded and compared against
all cached query embeddings. If the best match exceeds CACHE_THRESHOLD (0.95),
the stored answer + sources are returned immediately, skipping the pipeline.

After a pipeline run, the result is stored for future reuse with a 24-hour TTL.

Redis key schema:
  cache:embedding:{entry_id}  — float32 numpy bytes             TTL 24 h
  cache:answer:{entry_id}     — JSON {"answer": ..., "sources": [...]}  TTL 24 h
  cache:index                 — sorted set (member=entry_id, score=epoch)
                                used for LRU eviction at MAX_CACHE_ENTRIES
"""

import hashlib
import json
import logging
import time

import numpy as np
import redis.asyncio as aioredis

from backend.config import settings
from backend.services.retriever import _get_model

logger = logging.getLogger(__name__)

CACHE_TTL: int = 24 * 3600          # 24 hours in seconds
CACHE_THRESHOLD: float = 0.95       # minimum cosine similarity for a cache hit
MAX_CACHE_ENTRIES: int = 1000       # LRU eviction limit (~3 MB of embeddings)
_INDEX_KEY: str = "cache:index"

_redis_client: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis:
    """Return a cached async Redis client (connected lazily)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=False,  # we store raw bytes for embeddings
        )
    return _redis_client


def _embed(text: str) -> np.ndarray:
    """
    Embed text with the PubMedBERT singleton.

    Returns a normalized float32 numpy array. Normalisation means
    cosine similarity equals the dot product, which is fast to compute.
    """
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.astype(np.float32)


# ── Public API ─────────────────────────────────────────────────────────────────

async def check_cache(query: str) -> dict | None:
    """
    Check whether a semantically equivalent query is already cached.

    Embeds the query and computes cosine similarity (dot product of
    normalised vectors) against every cached embedding. Returns the
    cached ``{"answer": ..., "sources": [...]}`` dict when the best
    match meets or exceeds CACHE_THRESHOLD; otherwise returns None.

    Args:
        query: Raw user query string.

    Returns:
        Cached result dict on hit, None on miss.
    """
    redis = await _get_redis()
    entry_ids: list[bytes] = await redis.zrange(_INDEX_KEY, 0, -1)
    if not entry_ids:
        return None

    query_vec = _embed(query)

    best_similarity: float = 0.0
    best_entry_id: str | None = None

    for raw_id in entry_ids:
        entry_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
        emb_bytes: bytes | None = await redis.get(f"cache:embedding:{entry_id}")
        if emb_bytes is None:
            # Key expired between the index listing and this fetch — skip
            continue
        cached_vec = np.frombuffer(emb_bytes, dtype=np.float32)
        similarity = float(np.dot(query_vec, cached_vec))
        if similarity > best_similarity:
            best_similarity = similarity
            best_entry_id = entry_id

    if best_similarity >= CACHE_THRESHOLD and best_entry_id is not None:
        answer_json: bytes | None = await redis.get(f"cache:answer:{best_entry_id}")
        if answer_json:
            logger.info(
                "Cache HIT (similarity=%.3f) for query: %r",
                best_similarity,
                query[:60],
            )
            return json.loads(answer_json)

    logger.info(
        "Cache MISS (best=%.3f) for query: %r",
        best_similarity,
        query[:60],
    )
    return None


async def store_cache(query: str, answer: str, sources: list[dict]) -> None:
    """
    Persist a Q&A result in the semantic cache with a 24-hour TTL.

    The entry ID is an MD5 hash of the query string.  After inserting,
    the oldest entry is evicted if the total count exceeds MAX_CACHE_ENTRIES.

    Args:
        query:   Raw user query (used to produce the embedding and the key).
        answer:  Final generated answer string.
        sources: List of source citation dicts to cache alongside the answer.
    """
    redis = await _get_redis()
    entry_id = hashlib.md5(query.encode()).hexdigest()

    query_vec = _embed(query)

    await redis.set(f"cache:embedding:{entry_id}", query_vec.tobytes(), ex=CACHE_TTL)
    await redis.set(
        f"cache:answer:{entry_id}",
        json.dumps({"answer": answer, "sources": sources}),
        ex=CACHE_TTL,
    )
    await redis.zadd(_INDEX_KEY, {entry_id: time.time()})

    # ── Evict oldest entries if over the limit ─────────────────────────────────
    count: int = await redis.zcard(_INDEX_KEY)
    if count > MAX_CACHE_ENTRIES:
        n_to_evict = count - MAX_CACHE_ENTRIES
        to_evict: list[bytes] = await redis.zrange(_INDEX_KEY, 0, n_to_evict - 1)
        for raw_id in to_evict:
            old_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            await redis.delete(
                f"cache:embedding:{old_id}",
                f"cache:answer:{old_id}",
            )
        await redis.zremrangebyrank(_INDEX_KEY, 0, n_to_evict - 1)
        logger.info(
            "Cache evicted %d old entr%s (limit=%d)",
            n_to_evict,
            "y" if n_to_evict == 1 else "ies",
            MAX_CACHE_ENTRIES,
        )

    logger.info("Cache stored entry %s for query: %r", entry_id[:8], query[:60])


async def clear_cache() -> int:
    """
    Delete every semantic cache entry from Redis.

    Returns:
        Number of entries removed.
    """
    redis = await _get_redis()
    entry_ids: list[bytes] = await redis.zrange(_INDEX_KEY, 0, -1)
    count = len(entry_ids)

    for raw_id in entry_ids:
        entry_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
        await redis.delete(
            f"cache:embedding:{entry_id}",
            f"cache:answer:{entry_id}",
        )
    await redis.delete(_INDEX_KEY)

    logger.info("Cache cleared: %d entr%s removed", count, "y" if count == 1 else "ies")
    return count
