"""
MediBot v2 — RAGAS Evaluation & Retrieval Logging Service (Phase 10)

Two Celery tasks fired after every non-cached pipeline run:

  evaluate_with_ragas   — runs 6 RAGAS metrics (GPT-4o-mini as LLM judge)
                          and writes scores to retrieval_logs.ragas_scores
  log_retrieval_details — writes the full pipeline trace to retrieval_logs

Both tasks create their own short-lived SQLAlchemy async sessions so they
can run in a separate Celery worker process without sharing FastAPI's session.

Async code inside Celery tasks is wrapped with asyncio.run().  This is safe
because each Celery task runs in its own thread/process with no pre-existing
event loop.
"""

import asyncio
import logging
import math
import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.celery_app import celery_app
from backend.config import settings
from backend.models.database import RetrievalLog

logger = logging.getLogger(__name__)


# ── Database session factory for tasks ────────────────────────────────────────

async def _make_session() -> tuple[Any, AsyncSession]:
    """Create a minimal async engine + session for use inside a Celery task."""
    engine = create_async_engine(settings.database_url, pool_size=1, max_overflow=0)
    session = async_sessionmaker(engine, expire_on_commit=False)()
    return engine, session


# ── RAGAS helpers ──────────────────────────────────────────────────────────────

def _compute_ragas_scores(
    query: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """
    Run RAGAS evaluation synchronously and return a metric → score dict.

    Uses GPT-4o-mini as the LLM judge and OpenAI embeddings.
    Metrics: faithfulness, answer_relevancy, context_precision,
             context_recall, answer_correctness, answer_similarity.

    For context_recall, answer_correctness, and answer_similarity the
    generated answer is used as a ground-truth proxy (no real ground
    truth is available in production).

    Raises:
        RuntimeError: if the RAGAS call fails for any reason.
    """
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(f"RAGAS dependencies missing: {exc}") from exc

    # Build LLM and embeddings wrappers (RAGAS 0.2+ API)
    eval_kwargs: dict[str, Any] = {}
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        eval_kwargs["llm"] = LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.openai_model_aux,
                api_key=settings.openai_api_key,
                temperature=0.0,
            )
        )
        eval_kwargs["embeddings"] = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(api_key=settings.openai_api_key)
        )
    except ImportError:
        pass  # RAGAS 0.1.x — relies on OPENAI_API_KEY env var

    safe_contexts = contexts if contexts else ["No context available."]

    dataset = Dataset.from_dict(
        {
            "question": [query],
            "answer": [answer],
            "contexts": [safe_contexts],
            # Proxy ground truth: the model's own answer.
            # context_recall / answer_correctness / answer_similarity will
            # reflect self-consistency rather than factual accuracy, which
            # is still a useful production health signal.
            "ground_truth": [answer],
        }
    )

    try:
        result = ragas_evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
                answer_similarity,
            ],
            **eval_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"ragas.evaluate failed: {exc}") from exc

    raw: dict = result.to_pandas().iloc[0].to_dict()
    return {
        key: round(float(val), 4)
        for key, val in raw.items()
        if isinstance(val, (int, float)) and not math.isnan(float(val))
    }


# ── Async DB helpers (injectable db= for testing) ─────────────────────────────

async def _save_ragas_scores_to_db(
    message_id: str,
    scores: dict[str, float],
    db: AsyncSession | None = None,
) -> None:
    """
    UPDATE retrieval_logs SET ragas_scores = scores WHERE message_id = …

    If ``db`` is None, creates and disposes its own engine + session.
    """
    _own_session = db is None
    engine = None
    if _own_session:
        engine, db = await _make_session()

    try:
        await db.execute(
            sa.update(RetrievalLog)
            .where(RetrievalLog.message_id == uuid.UUID(message_id))
            .values(ragas_scores=scores)
        )
        await db.commit()
        logger.info(
            "RAGAS scores saved for message %s: %s",
            message_id[:8],
            {k: v for k, v in scores.items()},
        )
    finally:
        if _own_session and db is not None:
            await db.close()
        if engine is not None:
            await engine.dispose()


async def _create_retrieval_log(
    message_id: str,
    enhanced_query: str,
    hyde_answer: str,
    retrieved_child_ids: list[str],
    retrieved_parent_ids: list[str],
    reranker_scores: dict[str, float],
    relevance_verdict: str,
    hallucination_verdict: str,
    latency_ms: int,
    db: AsyncSession | None = None,
) -> None:
    """
    INSERT a new row into retrieval_logs with the full pipeline trace.

    If ``db`` is None, creates and disposes its own engine + session.
    """
    _own_session = db is None
    engine = None
    if _own_session:
        engine, db = await _make_session()

    def _to_uuids(id_strings: list[str]) -> list[uuid.UUID]:
        result: list[uuid.UUID] = []
        for s in id_strings:
            try:
                result.append(uuid.UUID(s))
            except (ValueError, AttributeError):
                pass
        return result

    log_row = RetrievalLog(
        id=uuid.uuid4(),
        message_id=uuid.UUID(message_id),
        enhanced_query=enhanced_query or None,
        hyde_answer=hyde_answer or None,
        retrieved_child_ids=_to_uuids(retrieved_child_ids) or None,
        retrieved_parent_ids=_to_uuids(retrieved_parent_ids) or None,
        reranker_scores=reranker_scores or None,
        relevance_verdict=relevance_verdict or None,
        hallucination_verdict=hallucination_verdict or None,
        latency_ms=latency_ms if latency_ms else None,
    )

    try:
        db.add(log_row)
        await db.commit()
        logger.info("Retrieval log created for message %s", message_id[:8])
    finally:
        if _own_session and db is not None:
            await db.close()
        if engine is not None:
            await engine.dispose()


# ── Celery tasks ───────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="evaluation.evaluate_with_ragas",
    max_retries=2,
    default_retry_delay=30,
)
def evaluate_with_ragas(
    self,
    message_id: str,
    query: str,
    answer: str,
    contexts: list[str],
) -> None:
    """
    Celery task: run RAGAS evaluation and write scores to retrieval_logs.

    Dispatched after every non-cached pipeline run.  Failures are logged
    but never surfaced to the user.

    Args:
        message_id: UUID string of the assistant chat_messages row.
        query:      Raw user query.
        answer:     Final generated answer.
        contexts:   List of compressed context strings fed to the generator.
    """
    try:
        scores = _compute_ragas_scores(query, answer, contexts)
    except Exception as exc:
        logger.error(
            "RAGAS evaluation failed for message %s: %s",
            message_id[:8],
            exc,
        )
        scores = {}

    asyncio.run(_save_ragas_scores_to_db(message_id, scores))


@celery_app.task(
    bind=True,
    name="evaluation.log_retrieval_details",
    max_retries=2,
    default_retry_delay=10,
)
def log_retrieval_details(
    self,
    message_id: str,
    enhanced_query: str,
    hyde_answer: str,
    retrieved_child_ids: list[str],
    retrieved_parent_ids: list[str],
    reranker_scores: dict[str, float],
    relevance_verdict: str,
    hallucination_verdict: str,
    latency_ms: int,
) -> None:
    """
    Celery task: write the full pipeline trace to retrieval_logs.

    Dispatched after every non-cached pipeline run alongside
    evaluate_with_ragas.  The retrieval_logs row is created here;
    evaluate_with_ragas later UPDATEs the ragas_scores column once
    evaluation completes.

    Args:
        message_id:            UUID string of the assistant chat_messages row.
        enhanced_query:        Rewritten query after memory + HyDE enhancement.
        hyde_answer:           Hypothetical answer used for HyDE retrieval.
        retrieved_child_ids:   Ordered child-chunk UUID strings after RRF.
        retrieved_parent_ids:  Parent-chunk UUID strings fetched from PostgreSQL.
        reranker_scores:       ``{chunk_id: rank}`` dict (rank 1 = best).
        relevance_verdict:     CRAG verdict (RELEVANT / PARTIAL / IRRELEVANT).
        hallucination_verdict: Hallucination guard verdict (grounded / ungrounded).
        latency_ms:            Total pipeline latency in milliseconds.
    """
    asyncio.run(
        _create_retrieval_log(
            message_id=message_id,
            enhanced_query=enhanced_query,
            hyde_answer=hyde_answer,
            retrieved_child_ids=retrieved_child_ids,
            retrieved_parent_ids=retrieved_parent_ids,
            reranker_scores=reranker_scores,
            relevance_verdict=relevance_verdict,
            hallucination_verdict=hallucination_verdict,
            latency_ms=latency_ms,
        )
    )
