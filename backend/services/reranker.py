"""
MediBot v2 — Re-ranking Service (Phase 5)

Two-stage re-ranking after Pinecone hybrid search:
  1. Reciprocal Rank Fusion (RRF) — merges 3 Pinecone result lists into one
     unified ranking using rank positions, not raw scores.
  2. Cross-encoder re-ranking — ms-marco-MiniLM-L-6-v2 scores each child chunk
     against the enhanced query; top-N chunks selected.
"""

import logging
from typing import Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
TOP_N = 5

_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    """Load the cross-encoder model once and reuse across requests."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading CrossEncoder: %s", CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
    return _cross_encoder


def rrf_merge(
    result_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[str]:
    """
    Reciprocal Rank Fusion over multiple Pinecone result lists.

    Each list is a sequence of Pinecone match dicts whose ``metadata`` field
    contains a ``child_id`` key.  Chunks appearing in multiple lists receive
    a cumulative bonus.

    Formula: score(d) = Σ 1 / (k + rank_i(d))  for each list i containing d

    Args:
        result_lists: Up to N lists of Pinecone match dicts (already ranked
                      by Pinecone score within each list).
        k:            Constant preventing division by zero and dampening the
                      impact of high-ranked chunks.  Default 60 per the paper.

    Returns:
        Child-id strings sorted by descending RRF score.
    """
    scores: dict[str, float] = {}

    for result_list in result_lists:
        for rank, match in enumerate(result_list, start=1):
            child_id = match.get("metadata", {}).get("child_id")
            if not child_id:
                continue
            scores[child_id] = scores.get(child_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    logger.info(
        "RRF produced %d unique child_ids from %d result lists",
        len(ranked),
        len(result_lists),
    )
    return ranked


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int = TOP_N,
) -> list[dict[str, Any]]:
    """
    Cross-encoder re-ranking of child chunks against the query.

    Each chunk must have a ``content`` key.  Chunks are scored as
    (query, chunk_content) pairs and the top-N by score are returned.

    Args:
        query:   Enhanced query string.
        chunks:  Child chunk dicts (``content`` required).
        top_n:   Number of highest-scoring chunks to keep.

    Returns:
        Up to ``top_n`` chunks sorted by descending cross-encoder score.
    """
    if not chunks:
        return []

    model = _get_cross_encoder()
    pairs = [(query, chunk["content"]) for chunk in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top = [chunk for _, chunk in ranked[:top_n]]
    logger.info(
        "Cross-encoder reranked %d chunks → top %d selected",
        len(chunks),
        len(top),
    )
    return top
