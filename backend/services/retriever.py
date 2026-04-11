"""
MediBot v2 — Retriever Service (Phase 3)

Performs hybrid search against Pinecone (dense + sparse in one call),
then fetches the corresponding parent chunks from PostgreSQL.
"""

import logging
import uuid
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.models.database import DocumentChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
BM25_PATH = "data/bm25_encoder.json"
TOP_K = 5

# ── Module-level singletons (loaded once per process) ─────────────────────────
_model: SentenceTransformer | None = None
_bm25: BM25Encoder | None = None
_pinecone_index = None


def _get_model() -> SentenceTransformer:
    """Load PubMedBERT once and reuse across requests."""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def _get_bm25() -> BM25Encoder | None:
    """
    Load the BM25 encoder saved during ingestion.
    Returns None if the encoder file doesn't exist yet (ingestion not run).
    """
    global _bm25
    if _bm25 is None:
        path = Path(BM25_PATH)
        if not path.exists():
            logger.warning(
                "BM25 encoder not found at '%s'. "
                "Run ingestion first. Falling back to dense-only search.",
                BM25_PATH,
            )
            return None
        logger.info("Loading BM25 encoder from %s", BM25_PATH)
        _bm25 = BM25Encoder()
        _bm25.load(path=BM25_PATH)
    return _bm25


def _get_pinecone_index():
    """Return a Pinecone Index handle (cached per process)."""
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _pinecone_index = pc.Index(settings.pinecone_index_name)
    return _pinecone_index


# ── Main retrieval function ────────────────────────────────────────────────────

async def retrieve(
    query: str,
    db: AsyncSession,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """
    Full hybrid retrieval pipeline:
      1. Embed query with PubMedBERT (dense, 768-dim)
      2. Encode query with BM25 (sparse) — skipped if encoder not available
      3. Single hybrid Pinecone query
      4. Extract parent_ids from result metadata
      5. Fetch parent chunk content from PostgreSQL
      6. Return list of parent chunk dicts with metadata

    Returns [] if Pinecone index doesn't exist or no results found.
    """
    # ── Step 1: Dense embedding ───────────────────────────────────────────────
    model = _get_model()
    dense_vector: list[float] = model.encode(
        [query], normalize_embeddings=True
    )[0].tolist()

    # ── Step 2: Sparse encoding ───────────────────────────────────────────────
    bm25 = _get_bm25()
    sparse_vector: dict | None = None
    if bm25 is not None:
        encoded = bm25.encode_queries([query])
        sparse_vector = encoded[0] if encoded else None

    # ── Step 3: Pinecone hybrid query ─────────────────────────────────────────
    try:
        index = _get_pinecone_index()
        query_kwargs: dict[str, Any] = {
            "vector": dense_vector,
            "top_k": top_k,
            "namespace": settings.pinecone_namespace,
            "include_metadata": True,
        }
        if sparse_vector:
            query_kwargs["sparse_vector"] = sparse_vector

        results = index.query(**query_kwargs)
    except Exception as exc:
        logger.error("Pinecone query failed: %s", exc)
        return []

    matches = results.get("matches", [])
    if not matches:
        logger.info("No Pinecone matches for query: %s", query[:80])
        return []

    logger.info("Pinecone returned %d matches", len(matches))

    # ── Step 4: Extract unique parent_ids from metadata ───────────────────────
    parent_ids: list[uuid.UUID] = []
    seen: set[str] = set()
    for match in matches:
        meta = match.get("metadata", {})
        pid_str = meta.get("parent_id")
        if pid_str and pid_str not in seen:
            seen.add(pid_str)
            try:
                parent_ids.append(uuid.UUID(pid_str))
            except ValueError:
                logger.warning("Invalid parent_id in Pinecone metadata: %s", pid_str)

    if not parent_ids:
        logger.warning("No valid parent_ids found in Pinecone results")
        return []

    # ── Step 5: Fetch parent chunks from PostgreSQL ───────────────────────────
    result = await db.execute(
        sa.select(DocumentChunk).where(
            DocumentChunk.id.in_(parent_ids),
            DocumentChunk.chunk_type == "parent",
        )
    )
    parent_rows = result.scalars().all()

    if not parent_rows:
        logger.warning("Parent chunks not found in PostgreSQL for ids: %s", parent_ids)
        return []

    # ── Step 6: Build return payload ──────────────────────────────────────────
    # Preserve Pinecone ranking order
    parent_map = {str(row.id): row for row in parent_rows}
    chunks: list[dict[str, Any]] = []

    for pid in parent_ids:
        row = parent_map.get(str(pid))
        if row:
            chunks.append(
                {
                    "chunk_id": str(row.id),
                    "content": row.content,
                    "page_number": row.page_number or 0,
                    "section_heading": row.section_heading or "",
                    "source_pdf": row.source_pdf,
                }
            )

    logger.info("Returning %d parent chunks for generation", len(chunks))
    return chunks
