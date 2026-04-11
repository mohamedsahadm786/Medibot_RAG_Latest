"""
Tests for backend/services/evaluation.py (Phase 10).
All DB, RAGAS, and Celery calls are mocked — no real PostgreSQL, OpenAI, or
broker usage.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.evaluation import (
    _compute_ragas_scores,
    _create_retrieval_log,
    _save_ragas_scores_to_db,
    evaluate_with_ragas,
    log_retrieval_details,
)

_MSG_ID = str(uuid.uuid4())


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_db_mock() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.close = AsyncMock()
    return session


# ── _save_ragas_scores_to_db ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_ragas_scores_executes_update_and_commits() -> None:
    """_save_ragas_scores_to_db must call execute + commit with the scores."""
    db = _make_db_mock()
    scores = {"faithfulness": 0.90, "answer_relevancy": 0.85}

    await _save_ragas_scores_to_db(_MSG_ID, scores, db=db)

    db.execute.assert_called_once()
    db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_save_ragas_scores_accepts_empty_scores() -> None:
    """Empty scores dict (RAGAS failed) should still execute + commit."""
    db = _make_db_mock()

    await _save_ragas_scores_to_db(_MSG_ID, {}, db=db)

    db.execute.assert_called_once()
    db.commit.assert_called_once()


# ── _create_retrieval_log ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_retrieval_log_adds_and_commits() -> None:
    """_create_retrieval_log must add a RetrievalLog row and commit."""
    db = _make_db_mock()

    await _create_retrieval_log(
        message_id=_MSG_ID,
        enhanced_query="What is diabetes?",
        hyde_answer="Diabetes is a condition…",
        retrieved_child_ids=[str(uuid.uuid4()), str(uuid.uuid4())],
        retrieved_parent_ids=[str(uuid.uuid4())],
        reranker_scores={"chunk-id-1": 1.0, "chunk-id-2": 2.0},
        relevance_verdict="RELEVANT",
        hallucination_verdict="grounded",
        latency_ms=420,
        db=db,
    )

    db.add.assert_called_once()
    db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_create_retrieval_log_skips_invalid_uuids() -> None:
    """Non-UUID strings in id lists must be silently skipped (no crash)."""
    db = _make_db_mock()

    await _create_retrieval_log(
        message_id=_MSG_ID,
        enhanced_query="query",
        hyde_answer="",
        retrieved_child_ids=["not-a-uuid", str(uuid.uuid4())],
        retrieved_parent_ids=["also-bad"],
        reranker_scores={},
        relevance_verdict="PARTIAL",
        hallucination_verdict="grounded",
        latency_ms=0,
        db=db,
    )

    db.add.assert_called_once()
    db.commit.assert_called_once()


# ── _compute_ragas_scores ──────────────────────────────────────────────────────

def test_compute_ragas_scores_returns_metric_dict() -> None:
    """_compute_ragas_scores should return a float dict on success."""
    # Build a fake RAGAS result whose .to_pandas().iloc[0].to_dict() returns scores
    fake_row = MagicMock()
    fake_row.to_dict.return_value = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.85,
        "context_precision": 0.8,
        "context_recall": 0.9,
        "answer_correctness": 0.88,
        "answer_similarity": 0.92,
    }
    fake_pandas = MagicMock()
    fake_pandas.iloc.__getitem__.return_value = fake_row

    fake_result = MagicMock()
    fake_result.to_pandas.return_value = fake_pandas

    # Patch sys.modules so the lazy `from ragas import evaluate` in
    # _compute_ragas_scores hits our mock, not the real library.
    mock_ragas = MagicMock()
    mock_ragas.evaluate = MagicMock(return_value=fake_result)

    mock_dataset_cls = MagicMock()
    mock_dataset_cls.from_dict.return_value = MagicMock()
    mock_datasets = MagicMock()
    mock_datasets.Dataset = mock_dataset_cls

    with patch.dict(
        "sys.modules",
        {
            "ragas": mock_ragas,
            "ragas.metrics": MagicMock(),
            "ragas.llms": MagicMock(),
            "ragas.embeddings": MagicMock(),
            "datasets": mock_datasets,
        },
    ):
        result = _compute_ragas_scores(
            "What is diabetes?",
            "Diabetes is a metabolic disease.",
            ["Context about diabetes."],
        )

    assert isinstance(result, dict)
    assert len(result) > 0
    assert all(isinstance(v, float) for v in result.values())


def test_compute_ragas_scores_raises_on_import_failure() -> None:
    """_compute_ragas_scores should raise RuntimeError when ragas is missing."""
    with patch.dict("sys.modules", {"ragas": None, "datasets": None}):
        with pytest.raises((RuntimeError, ImportError, TypeError)):
            _compute_ragas_scores("query", "answer", ["context"])


# ── Celery task: evaluate_with_ragas ──────────────────────────────────────────

def _run_and_close(coro):
    """Substitute for asyncio.run that closes the coroutine without running it."""
    coro.close()


def test_evaluate_with_ragas_saves_scores_on_success() -> None:
    """Task should call asyncio.run (to persist scores) on successful RAGAS eval."""
    scores = {"faithfulness": 0.9}

    with patch(
        "backend.services.evaluation._compute_ragas_scores",
        return_value=scores,
    ), patch(
        "backend.services.evaluation.asyncio.run",
        side_effect=_run_and_close,
    ) as mock_run:
        evaluate_with_ragas.run(
            message_id=_MSG_ID,
            query="What is diabetes?",
            answer="Diabetes is a metabolic disease.",
            contexts=["Context about diabetes."],
        )

    mock_run.assert_called_once()


def test_evaluate_with_ragas_saves_empty_dict_on_ragas_failure() -> None:
    """When RAGAS raises, asyncio.run should still be called with empty scores."""
    with patch(
        "backend.services.evaluation._compute_ragas_scores",
        side_effect=RuntimeError("RAGAS API error"),
    ), patch(
        "backend.services.evaluation.asyncio.run",
        side_effect=_run_and_close,
    ) as mock_run:
        evaluate_with_ragas.run(
            message_id=_MSG_ID,
            query="query",
            answer="answer",
            contexts=[],
        )

    mock_run.assert_called_once()


# ── Celery task: log_retrieval_details ────────────────────────────────────────

def test_log_retrieval_details_calls_asyncio_run() -> None:
    """log_retrieval_details should delegate to asyncio.run without raising."""
    with patch(
        "backend.services.evaluation.asyncio.run",
        side_effect=_run_and_close,
    ) as mock_run:
        log_retrieval_details.run(
            message_id=_MSG_ID,
            enhanced_query="What is diabetes?",
            hyde_answer="Diabetes is…",
            retrieved_child_ids=[str(uuid.uuid4())],
            retrieved_parent_ids=[str(uuid.uuid4())],
            reranker_scores={"chunk-1": 1.0},
            relevance_verdict="RELEVANT",
            hallucination_verdict="grounded",
            latency_ms=350,
        )

    mock_run.assert_called_once()
