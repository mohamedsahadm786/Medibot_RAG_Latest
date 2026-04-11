"""
MediBot v2 — Query Transformation Service (Phase 5)

Full pipeline before generation:
  1. Query Enhancement  — GPT-4o-mini rewrites raw query using conversation context
  2. HyDE              — Generate hypothetical answer, embed it, search (top_k=10)
  3. Multi-Query       — Generate 2 query variants, search each (top_k=5)
  4. RRF               — Reciprocal Rank Fusion over all 3 result lists
  5. Child Fetch        — Retrieve child chunk text from PostgreSQL
  6. Cross-Encoder     — Re-rank child chunks; keep top 5
  7. Parent Fetch       — Fetch unique parent chunks from PostgreSQL
  8. Compression        — GPT-4o-mini extracts relevant sentences per parent
"""

import asyncio
import logging
import uuid
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.services.compressor import compress_contexts
from backend.services.reranker import rerank_chunks, rrf_merge
from backend.services.retriever import (
    _encode_query,
    _fetch_child_chunks,
    _fetch_parent_chunks,
    _pinecone_search,
)

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

_ENHANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical query specialist. Rewrite the user's question into a "
            "specific, standalone medical query suitable for searching a medical encyclopedia.\n\n"
            "If conversation history is provided, use it to resolve any pronouns or "
            "references (e.g. 'it', 'that condition', 'those symptoms').\n"
            "Return ONLY the rewritten query — no explanation, no extra text.",
        ),
        (
            "human",
            "Conversation history:\n{summaries}\n\nRaw query: {query}\n\nRewritten query:",
        ),
    ]
)

_HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical encyclopedia author. Given a medical question, write a "
            "concise paragraph (~200 words) that would appear as the answer in a medical "
            "reference encyclopedia. Write factually and directly — no introductory phrases.",
        ),
        ("human", "Question: {query}"),
    ]
)

_VARIANTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical search expert. Given a medical query, generate exactly 2 "
            "alternative phrasings that would retrieve different but relevant information.\n"
            "Return ONLY the 2 queries, one per line, no numbering, no explanation.",
        ),
        ("human", "Original query: {query}"),
    ]
)

# ── LLM singleton (gpt-4o-mini, temp=0.3) ─────────────────────────────────────

_aux_llm: ChatOpenAI | None = None


def _get_aux_llm() -> ChatOpenAI:
    """Return a cached gpt-4o-mini instance for auxiliary LLM calls."""
    global _aux_llm
    if _aux_llm is None:
        _aux_llm = ChatOpenAI(
            model=settings.openai_model_aux,
            temperature=0.3,
            api_key=settings.openai_api_key,
        )
    return _aux_llm


# ── Individual transformation steps ───────────────────────────────────────────

async def enhance_query(raw_query: str, summaries: list[str]) -> str:
    """
    Rewrite raw_query into a specific, standalone medical query.

    Args:
        raw_query:  The user's original question.
        summaries:  Last N conversation turn summaries (empty list = first message).

    Returns:
        Enhanced query string, or raw_query if LLM returns empty.
    """
    summaries_text = (
        "\n".join(f"- {s}" for s in summaries) if summaries else "None"
    )
    llm = _get_aux_llm()
    chain = _ENHANCE_PROMPT | llm
    response = await chain.ainvoke({"summaries": summaries_text, "query": raw_query})
    enhanced = response.content.strip()
    logger.info("Query enhanced: %r -> %r", raw_query[:60], enhanced[:60])
    return enhanced or raw_query


async def _generate_hyde_text(enhanced_query: str) -> str:
    """Generate a hypothetical encyclopedia answer paragraph for HyDE."""
    llm = _get_aux_llm()
    chain = _HYDE_PROMPT | llm
    response = await chain.ainvoke({"query": enhanced_query})
    return response.content.strip()


async def _generate_variants(enhanced_query: str) -> list[str]:
    """
    Generate 2 alternative query phrasings.

    Falls back to the enhanced query itself if the LLM returns fewer than 2 lines.
    """
    llm = _get_aux_llm()
    chain = _VARIANTS_PROMPT | llm
    response = await chain.ainvoke({"query": enhanced_query})
    lines = [line.strip() for line in response.content.strip().splitlines() if line.strip()]
    variants = lines[:2]
    while len(variants) < 2:
        variants.append(enhanced_query)
    logger.info("Generated %d query variants", len(variants))
    return variants


# ── Main transform + retrieve ──────────────────────────────────────────────────

async def transform_and_retrieve(
    raw_query: str,
    summaries: list[str],
    db: AsyncSession,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Full query transformation + retrieval + re-ranking + compression pipeline:
      1. Enhance query using conversation summaries
      2. HyDE text + query variants generated concurrently
      3. Three Pinecone searches (HyDE top_k=10, var1 top_k=5, var2 top_k=5)
      4. RRF merge → ranked child_ids
      5. Fetch child chunks from PostgreSQL
      6. Cross-encoder rerank → top 5 child chunks
      7. Extract unique parent_ids; fetch parent chunks from PostgreSQL
      8. Contextual compression per parent chunk

    Args:
        raw_query:  Original user question.
        summaries:  Conversation history summaries (Phase 7 populates this).
        db:         Async SQLAlchemy session.

    Returns:
        (enhanced_query, list of compressed parent chunk dicts)
    """
    # Step 1: Enhance query
    enhanced_query = await enhance_query(raw_query, summaries)

    # Step 2: Generate HyDE text + query variants concurrently
    hyde_text, variants = await asyncio.gather(
        _generate_hyde_text(enhanced_query),
        _generate_variants(enhanced_query),
    )

    # Step 3: Encode all three texts
    hyde_dense, hyde_sparse = _encode_query(hyde_text)
    var1_dense, var1_sparse = _encode_query(variants[0])
    var2_dense, var2_sparse = _encode_query(variants[1])

    # Step 4: Three Pinecone searches
    hyde_matches = _pinecone_search(hyde_dense, hyde_sparse, top_k=10)
    var1_matches = _pinecone_search(var1_dense, var1_sparse, top_k=5)
    var2_matches = _pinecone_search(var2_dense, var2_sparse, top_k=5)

    logger.info(
        "Pinecone matches — HyDE: %d, var1: %d, var2: %d",
        len(hyde_matches),
        len(var1_matches),
        len(var2_matches),
    )

    # Step 5: RRF merge → ranked child_ids
    ranked_child_ids = rrf_merge([hyde_matches, var1_matches, var2_matches])

    if not ranked_child_ids:
        logger.info("No Pinecone matches — returning empty results")
        return enhanced_query, []

    # Step 6: Fetch child chunks from PostgreSQL (preserves RRF order)
    child_uuids: list[uuid.UUID] = []
    for cid_str in ranked_child_ids:
        try:
            child_uuids.append(uuid.UUID(cid_str))
        except ValueError:
            logger.warning("Invalid child_id from RRF: %s", cid_str)

    child_chunks = await _fetch_child_chunks(child_uuids, db)

    # Step 7: Cross-encoder rerank → top 5 child chunks
    top_children = rerank_chunks(enhanced_query, child_chunks, top_n=5)

    # Step 8: Extract unique parent_ids (preserving rerank order)
    parent_ids: list[uuid.UUID] = []
    seen: set[str] = set()
    for chunk in top_children:
        pid_str = chunk.get("parent_id")
        if pid_str and pid_str not in seen:
            seen.add(pid_str)
            try:
                parent_ids.append(uuid.UUID(pid_str))
            except ValueError:
                logger.warning("Invalid parent_id in child chunk: %s", pid_str)

    logger.info("Unique parent_ids from top children: %d", len(parent_ids))

    # Step 9: Fetch parent chunks from PostgreSQL
    parent_chunks = await _fetch_parent_chunks(parent_ids, db)

    # Step 10: Contextual compression
    compressed = await compress_contexts(enhanced_query, parent_chunks)

    logger.info(
        "transform_and_retrieve returning %d compressed chunks for: %s",
        len(compressed),
        enhanced_query[:80],
    )

    return enhanced_query, compressed
