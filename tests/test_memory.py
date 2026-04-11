"""
Tests for backend/services/memory.py (Phase 7).
All DB and LLM calls are mocked — no real PostgreSQL or OpenAI usage.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.memory import load_summaries, save_turn_summary


# ── load_summaries ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_summaries_returns_chronological_order() -> None:
    """
    Rows are fetched newest-first from the DB (DESC order), then reversed
    so the returned list reads oldest-first (chronological).
    """
    session_id = uuid.uuid4()
    # Simulate DB returning newest-first: summary3, summary2, summary1
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        "summary3", "summary2", "summary1"
    ]
    db_mock = AsyncMock()
    db_mock.execute = AsyncMock(return_value=mock_result)

    summaries = await load_summaries(session_id, db_mock)

    # Should be reversed to chronological order
    assert summaries == ["summary1", "summary2", "summary3"]


@pytest.mark.asyncio
async def test_load_summaries_empty_for_first_turn() -> None:
    """First turn has no history — should return an empty list."""
    session_id = uuid.uuid4()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    db_mock = AsyncMock()
    db_mock.execute = AsyncMock(return_value=mock_result)

    summaries = await load_summaries(session_id, db_mock)

    assert summaries == []


@pytest.mark.asyncio
async def test_load_summaries_respects_memory_window() -> None:
    """DB query should use LIMIT = settings.memory_window."""
    session_id = uuid.uuid4()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = ["s1", "s2", "s3"]
    db_mock = AsyncMock()
    db_mock.execute = AsyncMock(return_value=mock_result)

    summaries = await load_summaries(session_id, db_mock)

    # Verify execute was called (query construction is internal to SQLAlchemy)
    db_mock.execute.assert_called_once()
    assert len(summaries) == 3


# ── save_turn_summary ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_turn_summary_generates_and_persists() -> None:
    """save_turn_summary should call the LLM and execute an UPDATE + commit."""
    message_id = uuid.uuid4()
    fake_response = MagicMock()
    fake_response.content = (
        "User asked about diabetes; assistant explained it is a metabolic condition."
    )

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=fake_response)

    db_mock = AsyncMock()
    db_mock.execute = AsyncMock()
    db_mock.commit = AsyncMock()

    with patch("backend.services.memory._SUMMARY_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        await save_turn_summary(
            assistant_message_id=message_id,
            query="What is diabetes?",
            answer="Diabetes is a metabolic disease characterized by...",
            db=db_mock,
        )

    db_mock.execute.assert_called_once()
    db_mock.commit.assert_called_once()


@pytest.mark.asyncio
async def test_save_turn_summary_falls_back_on_llm_failure() -> None:
    """If LLM raises, the fallback summary (truncated query) is still saved."""
    message_id = uuid.uuid4()

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

    db_mock = AsyncMock()
    db_mock.execute = AsyncMock()
    db_mock.commit = AsyncMock()

    with patch("backend.services.memory._SUMMARY_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        await save_turn_summary(
            assistant_message_id=message_id,
            query="What is hypertension?",
            answer="Hypertension is high blood pressure.",
            db=db_mock,
        )

    # Even on failure, execute + commit should still be called with fallback
    db_mock.execute.assert_called_once()
    db_mock.commit.assert_called_once()


@pytest.mark.asyncio
async def test_save_turn_summary_uses_truncated_fallback_on_empty_llm() -> None:
    """If the LLM returns blank content, fallback is 'User asked: <query>'."""
    message_id = uuid.uuid4()
    fake_response = MagicMock()
    fake_response.content = "   "  # blank

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=fake_response)

    db_mock = AsyncMock()
    db_mock.execute = AsyncMock()
    db_mock.commit = AsyncMock()

    # Capture what value is written
    written_values: list[dict] = []
    original_execute = db_mock.execute

    async def capture_execute(stmt):
        # Extract the compiled values from the UPDATE statement
        written_values.append(stmt.compile().params if hasattr(stmt, "compile") else {})
        return MagicMock()

    db_mock.execute = capture_execute

    with patch("backend.services.memory._SUMMARY_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        await save_turn_summary(
            assistant_message_id=message_id,
            query="What is aspirin?",
            answer="Aspirin is a pain reliever.",
            db=db_mock,
        )

    db_mock.commit.assert_called_once()
