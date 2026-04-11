"""
Tests for backend/services/compressor.py (Phase 5).
All LLM calls are mocked — no real OpenAI API usage.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.compressor import compress_contexts


# ── compress_contexts ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compress_contexts_extracts_relevant_sentences() -> None:
    """compress_contexts should replace content with the LLM's extracted text."""
    chunk = {
        "chunk_id": "parent-001",
        "content": "Diabetes is a metabolic disease. Unrelated sentence. Insulin is key.",
        "page_number": 5,
        "section_heading": "Diabetes",
        "source_pdf": "Medical_book.pdf",
    }
    fake_response = MagicMock()
    fake_response.content = "Diabetes is a metabolic disease. Insulin is key."

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=fake_response)

    with patch("backend.services.compressor._COMPRESS_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        result = await compress_contexts("What is diabetes?", [chunk])

    assert len(result) == 1
    assert result[0]["content"] == "Diabetes is a metabolic disease. Insulin is key."
    # Other fields should be preserved
    assert result[0]["chunk_id"] == "parent-001"
    assert result[0]["page_number"] == 5


@pytest.mark.asyncio
async def test_compress_contexts_falls_back_on_empty_llm_response() -> None:
    """If the LLM returns blank text, the original content is kept."""
    chunk = {
        "chunk_id": "parent-002",
        "content": "Original content here.",
        "page_number": 1,
        "section_heading": "Intro",
        "source_pdf": "Medical_book.pdf",
    }
    fake_response = MagicMock()
    fake_response.content = "   "  # blank

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=fake_response)

    with patch("backend.services.compressor._COMPRESS_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        result = await compress_contexts("some query", [chunk])

    assert result[0]["content"] == "Original content here."


@pytest.mark.asyncio
async def test_compress_contexts_empty_chunks_returns_empty() -> None:
    """Empty input should return empty list without calling the LLM."""
    result = await compress_contexts("query", [])
    assert result == []


@pytest.mark.asyncio
async def test_compress_contexts_llm_failure_falls_back_to_original() -> None:
    """On LLM exception, the original chunk is kept and processing continues."""
    chunk = {
        "chunk_id": "parent-003",
        "content": "Original text.",
        "page_number": 2,
        "section_heading": "Section",
        "source_pdf": "Medical_book.pdf",
    }

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    with patch("backend.services.compressor._COMPRESS_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        result = await compress_contexts("query", [chunk])

    assert len(result) == 1
    assert result[0]["content"] == "Original text."


@pytest.mark.asyncio
async def test_compress_contexts_processes_multiple_chunks() -> None:
    """Each chunk should be compressed independently; all are returned."""
    chunks = [
        {"chunk_id": f"p-{i}", "content": f"Content {i}.", "page_number": i,
         "section_heading": "S", "source_pdf": "book.pdf"}
        for i in range(3)
    ]

    call_count = 0

    async def fake_ainvoke(inputs: dict) -> MagicMock:
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.content = f"Compressed {call_count}."
        return resp

    mock_chain = MagicMock()
    mock_chain.ainvoke = fake_ainvoke

    with patch("backend.services.compressor._COMPRESS_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        result = await compress_contexts("query", chunks)

    assert len(result) == 3
    assert call_count == 3
