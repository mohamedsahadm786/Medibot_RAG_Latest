"""
MediBot v2 — Query Transformation Service (Phase 4)

Three-stage pipeline before Pinecone retrieval:
  1. Query Enhancement  — GPT-4o-mini rewrites raw query using conversation context
  2. HyDE              — Generate hypothetical answer, embed it, search (top_k=10)
  3. Multi-Query       — Generate 2 query variants, search each (top_k=5)

Results from all three searches are merged and deduplicated before fetching
parent chunks from PostgreSQL.
"""

import asyncio
import logging
import uuid
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.services.retriever import (
    _encode_query,
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
    Full query transformation + retrieval pipeline:
      1. Enhance query using conversation summaries
      2. HyDE + multi-query variants generated concurrently
      3. Three Pinecone searches (HyDE top_k=10, var1 top_k=5, var2 top_k=5)
      4. Deduplicate parent_ids (HyDE results ranked first)
      5. Fetch parent chunks from PostgreSQL

    Args:
        raw_query:  Original user question.
        summaries:  Conversation history summaries (Phase 7 populates this).
        db:         Async SQLAlchemy session.

    Returns:
        (enhanced_query, list of parent chunk dicts)
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

    # Step 5: Deduplicate parent_ids (HyDE-ranked first)
    all_matches = hyde_matches + var1_matches + var2_matches
    parent_ids: list[uuid.UUID] = []
    seen: set[str] = set()

    for match in all_matches:
        meta = match.get("metadata", {})
        pid_str = meta.get("parent_id")
        if pid_str and pid_str not in seen:
            seen.add(pid_str)
            try:
                parent_ids.append(uuid.UUID(pid_str))
            except ValueError:
                logger.warning("Invalid parent_id in Pinecone metadata: %s", pid_str)

    logger.info("Unique parent_ids after dedup: %d", len(parent_ids))

    # Step 6: Fetch parent chunks from PostgreSQL
    chunks = await _fetch_parent_chunks(parent_ids, db)
    logger.info(
        "transform_and_retrieve returning %d chunks for query: %s",
        len(chunks),
        enhanced_query[:80],
    )

    return enhanced_query, chunks
