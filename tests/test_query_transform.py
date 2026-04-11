"""
Tests for backend/services/query_transform.py (Phase 5).
All LLM, Pinecone, and cross-encoder calls are mocked — no real API usage.
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
    transform_and_retrieve should run the full pipeline:
    enhance → hyde → variants → pinecone x3 → RRF → fetch children →
    rerank → fetch parents → compress, returning compressed parent chunks.
    """
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())

    fake_child = {
        "chunk_id": child_id,
        "parent_id": parent_id,
        "content": "Diabetes content...",
        "page_number": 10,
        "section_heading": "Diabetes",
        "source_pdf": "Medical_book.pdf",
    }
    fake_parent = {
        "chunk_id": parent_id,
        "content": "Diabetes content...",
        "page_number": 10,
        "section_heading": "Diabetes",
        "source_pdf": "Medical_book.pdf",
    }
    fake_compressed = {**fake_parent, "content": "Compressed: Diabetes content."}

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
        return_value=[{"metadata": {"child_id": child_id, "parent_id": parent_id}}],
    ), patch(
        "backend.services.query_transform.rrf_merge",
        return_value=[child_id],
    ), patch(
        "backend.services.query_transform._fetch_child_chunks",
        new=AsyncMock(return_value=[fake_child]),
    ), patch(
        "backend.services.query_transform.rerank_chunks",
        return_value=[fake_child],
    ), patch(
        "backend.services.query_transform._fetch_parent_chunks",
        new=AsyncMock(return_value=[fake_parent]),
    ), patch(
        "backend.services.query_transform.compress_contexts",
        new=AsyncMock(return_value=[fake_compressed]),
    ):
        enhanced_query, chunks = await transform_and_retrieve(
            raw_query="what is diabetes?",
            summaries=[],
            db=db_mock,
        )

    assert enhanced_query == "symptoms of diabetes mellitus"
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == parent_id
    assert chunks[0]["content"] == "Compressed: Diabetes content."


@pytest.mark.asyncio
async def test_transform_and_retrieve_deduplicates_parent_ids() -> None:
    """
    When the top reranked children share a parent_id, only one parent is fetched.
    """
    shared_parent_id = str(uuid.uuid4())
    child_id_1 = str(uuid.uuid4())
    child_id_2 = str(uuid.uuid4())

    child_1 = {"chunk_id": child_id_1, "parent_id": shared_parent_id, "content": "A..."}
    child_2 = {"chunk_id": child_id_2, "parent_id": shared_parent_id, "content": "B..."}

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
        return_value=[],
    ), patch(
        "backend.services.query_transform.rrf_merge",
        return_value=[child_id_1, child_id_2],
    ), patch(
        "backend.services.query_transform._fetch_child_chunks",
        new=AsyncMock(return_value=[child_1, child_2]),
    ), patch(
        "backend.services.query_transform.rerank_chunks",
        return_value=[child_1, child_2],  # both map to same parent
    ), patch(
        "backend.services.query_transform._fetch_parent_chunks",
        new=AsyncMock(return_value=[{"chunk_id": shared_parent_id, "content": "Parent..."}]),
    ) as mock_fetch_parents, patch(
        "backend.services.query_transform.compress_contexts",
        new=AsyncMock(return_value=[{"chunk_id": shared_parent_id, "content": "Compressed..."}]),
    ):
        _, chunks = await transform_and_retrieve("query", [], db_mock)

    # Only 1 unique parent_id should be fetched
    fetched_ids = mock_fetch_parents.call_args[0][0]
    assert len(fetched_ids) == 1
    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_transform_and_retrieve_empty_pinecone_returns_empty() -> None:
    """If RRF returns no child_ids, result is empty list with no PG calls."""
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
        "backend.services.query_transform.rrf_merge",
        return_value=[],
    ), patch(
        "backend.services.query_transform._fetch_child_chunks",
        new=AsyncMock(return_value=[]),
    ) as mock_fetch_children:
        enhanced_query, chunks = await transform_and_retrieve("query", [], db_mock)

    assert chunks == []
    assert enhanced_query == "enhanced query"
    # Early return — child fetch should not be called when RRF returns nothing
    mock_fetch_children.assert_not_called()
