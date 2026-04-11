"""
Tests for backend/services/query_transform.py (Phase 4).
All LLM and Pinecone calls are mocked — no real API usage.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from backend.services.query_transform import (
    enhance_query,
    transform_and_retrieve,
)


# ── enhance_query ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enhance_query_no_history() -> None:
    """With no summaries, enhance_query should return the LLM's rewritten query."""
    fake_response = MagicMock()
    fake_response.content = "What are the symptoms and treatment of diabetes mellitus?"

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=fake_response)

    # Patch the module-level prompt so _ENHANCE_PROMPT | llm returns our mock chain
    with patch("backend.services.query_transform._ENHANCE_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        result = await enhance_query("what is diabetes?", summaries=[])

    assert result == "What are the symptoms and treatment of diabetes mellitus?"


@pytest.mark.asyncio
async def test_enhance_query_falls_back_to_raw_on_empty_response() -> None:
    """If LLM returns empty string, enhance_query returns the raw query."""
    fake_response = MagicMock()
    fake_response.content = "   "  # blank

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=fake_response)

    with patch("backend.services.query_transform._ENHANCE_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        result = await enhance_query("what is diabetes?", summaries=[])

    assert result == "what is diabetes?"


# ── transform_and_retrieve ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transform_and_retrieve_returns_enhanced_query_and_chunks() -> None:
    """
    transform_and_retrieve should call enhance, hyde, variants, search Pinecone
    3 times, deduplicate, and fetch parent chunks.
    All external calls are mocked.
    """
    parent_id = str(uuid.uuid4())
    fake_match = {"metadata": {"parent_id": parent_id, "child_id": str(uuid.uuid4())}}
    fake_chunk = {
        "chunk_id": parent_id,
        "content": "Diabetes content...",
        "page_number": 10,
        "section_heading": "Diabetes",
        "source_pdf": "Medical_book.pdf",
    }

    db_mock = AsyncMock()

    with patch(
        "backend.services.query_transform.enhance_query",
        new=AsyncMock(return_value="symptoms of diabetes mellitus"),
    ), patch(
        "backend.services.query_transform._generate_hyde_text",
        new=AsyncMock(return_value="Hypothetical encyclopedia answer about diabetes..."),
    ), patch(
        "backend.services.query_transform._generate_variants",
        new=AsyncMock(return_value=["diabetes symptoms", "diabetes treatment"]),
    ), patch(
        "backend.services.query_transform._encode_query",
        return_value=([0.1] * 768, {"indices": [1], "values": [0.5]}),
    ), patch(
        "backend.services.query_transform._pinecone_search",
        return_value=[fake_match],
    ), patch(
        "backend.services.query_transform._fetch_parent_chunks",
        new=AsyncMock(return_value=[fake_chunk]),
    ):
        enhanced_query, chunks = await transform_and_retrieve(
            raw_query="what is diabetes?",
            summaries=[],
            db=db_mock,
        )

    assert enhanced_query == "symptoms of diabetes mellitus"
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == parent_id


@pytest.mark.asyncio
async def test_transform_and_retrieve_deduplicates_parent_ids() -> None:
    """
    When HyDE and variant searches return the same parent_id,
    only one chunk should be fetched.
    """
    shared_parent_id = str(uuid.uuid4())
    fake_match = {"metadata": {"parent_id": shared_parent_id}}

    db_mock = AsyncMock()

    with patch(
        "backend.services.query_transform.enhance_query",
        new=AsyncMock(return_value="enhanced query"),
    ), patch(
        "backend.services.query_transform._generate_hyde_text",
        new=AsyncMock(return_value="hyde text"),
    ), patch(
        "backend.services.query_transform._generate_variants",
        new=AsyncMock(return_value=["variant 1", "variant 2"]),
    ), patch(
        "backend.services.query_transform._encode_query",
        return_value=([0.1] * 768, None),
    ), patch(
        "backend.services.query_transform._pinecone_search",
        return_value=[fake_match],   # all 3 searches return same match
    ) as mock_search, patch(
        "backend.services.query_transform._fetch_parent_chunks",
        new=AsyncMock(return_value=[{"chunk_id": shared_parent_id}]),
    ) as mock_fetch:
        _, chunks = await transform_and_retrieve("query", [], db_mock)

    # Pinecone searched 3 times
    assert mock_search.call_count == 3
    # But fetch called with only 1 unique parent_id
    fetched_ids = mock_fetch.call_args[0][0]
    assert len(fetched_ids) == 1


@pytest.mark.asyncio
async def test_transform_and_retrieve_empty_pinecone_returns_empty() -> None:
    """If all Pinecone searches return no matches, result is empty list."""
    db_mock = AsyncMock()

    with patch(
        "backend.services.query_transform.enhance_query",
        new=AsyncMock(return_value="enhanced query"),
    ), patch(
        "backend.services.query_transform._generate_hyde_text",
        new=AsyncMock(return_value="hyde text"),
    ), patch(
        "backend.services.query_transform._generate_variants",
        new=AsyncMock(return_value=["v1", "v2"]),
    ), patch(
        "backend.services.query_transform._encode_query",
        return_value=([0.1] * 768, None),
    ), patch(
        "backend.services.query_transform._pinecone_search",
        return_value=[],
    ), patch(
        "backend.services.query_transform._fetch_parent_chunks",
        new=AsyncMock(return_value=[]),
    ):
        enhanced_query, chunks = await transform_and_retrieve("query", [], db_mock)

    assert chunks == []
    assert enhanced_query == "enhanced query"
