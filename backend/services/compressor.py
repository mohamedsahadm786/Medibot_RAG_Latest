"""
MediBot v2 — Contextual Compression Service (Phase 5)

Uses GPT-4o-mini to extract only the sentences in each parent chunk
that are directly relevant to the user's enhanced query.  Shorter,
more focused contexts improve generation quality.
"""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.config import settings

logger = logging.getLogger(__name__)

_COMPRESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical text extractor. Given a passage from a medical encyclopedia "
            "and a user query, extract ONLY the sentences that are directly relevant to "
            "answering the query. Return the extracted sentences verbatim, separated by a "
            "single space. If nothing is clearly relevant, return the first two sentences "
            "of the passage.",
        ),
        (
            "human",
            "Query: {query}\n\nPassage:\n{passage}\n\nRelevant sentences:",
        ),
    ]
)

_compress_llm: ChatOpenAI | None = None


def _get_compress_llm() -> ChatOpenAI:
    """Return a cached GPT-4o-mini instance for compression calls."""
    global _compress_llm
    if _compress_llm is None:
        _compress_llm = ChatOpenAI(
            model=settings.openai_model_aux,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )
    return _compress_llm


async def compress_contexts(
    query: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compress each parent chunk to sentences relevant to the query.

    For each chunk, GPT-4o-mini extracts the most relevant sentences.
    On LLM failure or empty response, the original chunk content is kept.

    Args:
        query:   Enhanced query string.
        chunks:  Parent chunk dicts (must include ``content`` key).

    Returns:
        Chunk dicts with ``content`` replaced by the compressed text.
        Order and all other fields are preserved.
    """
    if not chunks:
        return []

    llm = _get_compress_llm()
    chain = _COMPRESS_PROMPT | llm

    compressed: list[dict[str, Any]] = []
    for chunk in chunks:
        try:
            response = await chain.ainvoke(
                {"query": query, "passage": chunk["content"]}
            )
            compressed_text = response.content.strip()
            if not compressed_text:
                compressed_text = chunk["content"]
            compressed.append({**chunk, "content": compressed_text})
            logger.debug(
                "Compressed chunk %s: %d → %d chars",
                chunk.get("chunk_id", "?"),
                len(chunk["content"]),
                len(compressed_text),
            )
        except Exception as exc:
            logger.warning(
                "Compression failed for chunk %s: %s",
                chunk.get("chunk_id", "?"),
                exc,
            )
            compressed.append(chunk)

    logger.info("Compressed %d parent chunks", len(compressed))
    return compressed
