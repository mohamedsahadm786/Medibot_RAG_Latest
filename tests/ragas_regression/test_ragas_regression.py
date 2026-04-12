"""
MediBot v2 — RAGAS Regression Test Suite (Phase 14)

Runs the full RAG pipeline against a curated set of medical Q&A pairs
and asserts that average faithfulness and answer_relevancy stay above
defined thresholds.

Usage:
    pytest tests/ragas_regression/test_ragas_regression.py -v -s

Or via the GitHub Actions manual workflow (ragas_regression.yml).

Results are written to tests/ragas_regression/results.json after each run.
"""

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# ── Paths and thresholds ───────────────────────────────────────────────────────

TEST_SUITE_PATH = Path(__file__).parent / "test_suite.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

FAITHFULNESS_THRESHOLD = 0.85
RELEVANCY_THRESHOLD = 0.80


# ── DB session factory ─────────────────────────────────────────────────────────

async def _make_db_session():
    """Create a minimal async SQLAlchemy session for the regression runner."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from backend.config import settings

    engine = create_async_engine(settings.database_url, pool_size=1, max_overflow=0)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory()


# ── Pipeline runner ────────────────────────────────────────────────────────────

async def _run_pipeline_for_question(question: str) -> dict[str, Any]:
    """
    Run the full RAG pipeline for a single question and return
    the answer, sources, and compressed contexts.
    """
    from backend.services.rag_pipeline import GraphState, build_pipeline

    engine, db = await _make_db_session()
    try:
        initial_state: GraphState = {
            "query": question,
            "session_id": "ragas-regression",
            "summaries": [],
            "intent": "",
            "enhanced_query": "",
            "hyde_answer": "",
            "query_variants": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "parent_chunks": [],
            "compressed_contexts": [],
            "relevance_verdict": "",
            "answer": "",
            "sources": [],
            "hallucination_verdict": "",
            "retry_count": 0,
            "hallucination_retry": False,
            "status_events": [],
        }
        pipeline = build_pipeline(db)
        final_state: GraphState = await pipeline.ainvoke(initial_state)
        return {
            "answer": final_state["answer"],
            "contexts": [c.get("content", "") for c in final_state["compressed_contexts"]],
            "sources": final_state["sources"],
        }
    finally:
        await db.close()
        await engine.dispose()


# ── RAGAS scorer ───────────────────────────────────────────────────────────────

def _score_with_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float]:
    """Run faithfulness + answer_relevancy via RAGAS and return scores."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate as ragas_evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness
    from backend.config import settings

    llm = LangchainLLMWrapper(
        ChatOpenAI(model=settings.openai_model_aux, api_key=settings.openai_api_key, temperature=0.0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(api_key=settings.openai_api_key)
    )

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts if contexts else ["No context available."]],
        "ground_truth": [ground_truth],
    })

    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )
    raw: dict = result.to_pandas().iloc[0].to_dict()
    return {
        key: round(float(val), 4)
        for key, val in raw.items()
        if isinstance(val, (int, float)) and not math.isnan(float(val))
    }


# ── Main regression test ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ragas_regression() -> None:
    """
    Run the full RAGAS regression suite.

    Loads test_suite.json, runs each question through the RAG pipeline,
    scores with RAGAS, asserts average thresholds, and writes results.json.
    """
    if not TEST_SUITE_PATH.exists():
        pytest.skip("test_suite.json not found — populate it before running regression.")

    test_cases: list[dict] = json.loads(TEST_SUITE_PATH.read_text(encoding="utf-8"))

    if not test_cases:
        pytest.skip("test_suite.json is empty — add Q&A pairs before running regression.")

    results = []
    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []

    for i, case in enumerate(test_cases, start=1):
        question = case["question"]
        expected_answer = case.get("expected_answer", "")
        expected_page = case.get("expected_source_page")

        logger.info("[%d/%d] Running: %s", i, len(test_cases), question[:80])

        # Run RAG pipeline
        try:
            pipeline_result = await _run_pipeline_for_question(question)
        except Exception as exc:
            logger.error("Pipeline failed for question %d: %s", i, exc)
            results.append({
                "question": question,
                "error": str(exc),
                "passed": False,
            })
            continue

        answer = pipeline_result["answer"]
        contexts = pipeline_result["contexts"]
        sources = pipeline_result["sources"]

        # Run RAGAS
        try:
            scores = _score_with_ragas(question, answer, contexts, expected_answer)
        except Exception as exc:
            logger.error("RAGAS failed for question %d: %s", i, exc)
            scores = {}

        # Check source page if expected
        source_pages = [s.get("page_number") for s in sources]
        page_found = (expected_page is None) or (expected_page in source_pages)

        faithfulness_val = scores.get("faithfulness", 0.0)
        relevancy_val = scores.get("answer_relevancy", 0.0)

        if faithfulness_val:
            faithfulness_scores.append(faithfulness_val)
        if relevancy_val:
            relevancy_scores.append(relevancy_val)

        results.append({
            "question": question,
            "answer": answer,
            "expected_source_page": expected_page,
            "source_pages_found": source_pages,
            "page_check_passed": page_found,
            "ragas_scores": scores,
            "passed": faithfulness_val >= FAITHFULNESS_THRESHOLD
                      and relevancy_val >= RELEVANCY_THRESHOLD,
        })

        logger.info(
            "  faithfulness=%.2f  relevancy=%.2f  page_ok=%s",
            faithfulness_val, relevancy_val, page_found,
        )

    # ── Compute averages ───────────────────────────────────────────────────────
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0
    passed_count = sum(1 for r in results if r.get("passed"))

    summary = {
        "total": len(test_cases),
        "passed": passed_count,
        "failed": len(test_cases) - passed_count,
        "avg_faithfulness": round(avg_faithfulness, 4),
        "avg_answer_relevancy": round(avg_relevancy, 4),
        "thresholds": {
            "faithfulness": FAITHFULNESS_THRESHOLD,
            "answer_relevancy": RELEVANCY_THRESHOLD,
        },
        "results": results,
    }

    # Write results.json
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Regression complete — faithfulness=%.2f (threshold %.2f), "
        "relevancy=%.2f (threshold %.2f), passed=%d/%d",
        avg_faithfulness, FAITHFULNESS_THRESHOLD,
        avg_relevancy, RELEVANCY_THRESHOLD,
        passed_count, len(test_cases),
    )

    # ── Assert thresholds ──────────────────────────────────────────────────────
    assert avg_faithfulness >= FAITHFULNESS_THRESHOLD, (
        f"Avg faithfulness {avg_faithfulness:.2f} below threshold {FAITHFULNESS_THRESHOLD}. "
        f"See {RESULTS_PATH} for details."
    )
    assert avg_relevancy >= RELEVANCY_THRESHOLD, (
        f"Avg answer_relevancy {avg_relevancy:.2f} below threshold {RELEVANCY_THRESHOLD}. "
        f"See {RESULTS_PATH} for details."
    )
