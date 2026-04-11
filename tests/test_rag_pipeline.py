"""
Tests for backend/services/rag_pipeline.py (Phase 6).
All LLM, Pinecone, cross-encoder, and DB calls are mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from backend.services.rag_pipeline import GraphState, build_pipeline


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_initial_state(**overrides) -> GraphState:
    """Return a fully populated initial GraphState with safe defaults."""
    base: GraphState = {
        "query": "what is diabetes?",
        "session_id": "test-session",
        "summaries": [],
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
    base.update(overrides)
    return base


def _child_chunk(child_id: str, parent_id: str, content: str = "Medical content.") -> dict:
    return {
        "chunk_id": child_id,
        "parent_id": parent_id,
        "content": content,
        "page_number": 1,
        "section_heading": "Test",
        "source_pdf": "book.pdf",
    }


def _parent_chunk(parent_id: str, content: str = "Parent content.") -> dict:
    return {
        "chunk_id": parent_id,
        "content": content,
        "page_number": 1,
        "section_heading": "Test",
        "source_pdf": "book.pdf",
    }


# ── Full happy-path test ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_happy_path_relevant_grounded() -> None:
    """
    Full graph run: RELEVANT verdict + grounded hallucination check.
    Should complete in a single pass without retries.
    """
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())

    fake_child = _child_chunk(child_id, parent_id)
    fake_parent = _parent_chunk(parent_id)
    fake_compressed = {**fake_parent, "content": "Compressed: Diabetes."}

    relevance_response = MagicMock()
    relevance_response.content = "RELEVANT"
    hallucination_response = MagicMock()
    hallucination_response.content = "grounded"

    mock_llm = MagicMock()
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=[relevance_response, hallucination_response])
    mock_llm.__or__ = MagicMock(return_value=mock_chain)

    db_mock = AsyncMock()

    with patch("backend.services.rag_pipeline.enhance_query",
               new=AsyncMock(return_value="symptoms of diabetes mellitus")), \
         patch("backend.services.rag_pipeline._generate_hyde_text",
               new=AsyncMock(return_value="HyDE answer about diabetes.")), \
         patch("backend.services.rag_pipeline._generate_variants",
               new=AsyncMock(return_value=["diabetes symptoms", "diabetes treatment"])), \
         patch("backend.services.rag_pipeline._encode_query",
               return_value=([0.1] * 768, None)), \
         patch("backend.services.rag_pipeline._pinecone_search",
               return_value=[{"metadata": {"child_id": child_id}}]), \
         patch("backend.services.rag_pipeline.rrf_merge",
               return_value=[child_id]), \
         patch("backend.services.rag_pipeline._fetch_child_chunks",
               new=AsyncMock(return_value=[fake_child])), \
         patch("backend.services.rag_pipeline.rerank_chunks",
               return_value=[fake_child]), \
         patch("backend.services.rag_pipeline._fetch_parent_chunks",
               new=AsyncMock(return_value=[fake_parent])), \
         patch("backend.services.rag_pipeline.compress_contexts",
               new=AsyncMock(return_value=[fake_compressed])), \
         patch("backend.services.rag_pipeline._RELEVANCE_PROMPT") as mock_rel_prompt, \
         patch("backend.services.rag_pipeline._HALLUCINATION_PROMPT") as mock_hal_prompt, \
         patch("backend.services.rag_pipeline.generate",
               new=AsyncMock(return_value={
                   "answer": "Diabetes is a metabolic condition.",
                   "sources": [{"chunk_id": parent_id, "page_number": 1,
                                "section_heading": "Test", "excerpt": "...",
                                "source_pdf": "book.pdf"}],
               })):

        mock_rel_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_hal_prompt.__or__ = MagicMock(return_value=mock_chain)

        pipeline = build_pipeline(db_mock)
        result = await pipeline.ainvoke(_make_initial_state())

    assert result["answer"] == "Diabetes is a metabolic condition."
    assert result["enhanced_query"] == "symptoms of diabetes mellitus"
    assert result["relevance_verdict"] == "RELEVANT"
    assert result["hallucination_verdict"] == "grounded"
    assert result["retry_count"] == 0
    assert result["hallucination_retry"] is False
    assert len(result["sources"]) == 1


# ── CRAG retry test ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_crag_retry_on_irrelevant() -> None:
    """
    When check_relevance returns IRRELEVANT on the first pass, the pipeline
    loops back to enhance_query (retry_count=1), then generates on second pass.
    """
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    fake_child = _child_chunk(child_id, parent_id)
    fake_parent = _parent_chunk(parent_id)
    fake_compressed = {**fake_parent, "content": "Compressed."}

    # First relevance check: IRRELEVANT → retry; second: RELEVANT → proceed
    relevance_call_count = 0

    async def relevance_ainvoke(inputs: dict) -> MagicMock:
        nonlocal relevance_call_count
        relevance_call_count += 1
        resp = MagicMock()
        resp.content = "IRRELEVANT" if relevance_call_count == 1 else "RELEVANT"
        return resp

    hallucination_response = MagicMock()
    hallucination_response.content = "grounded"

    mock_rel_chain = MagicMock()
    mock_rel_chain.ainvoke = relevance_ainvoke

    mock_hal_chain = MagicMock()
    mock_hal_chain.ainvoke = AsyncMock(return_value=hallucination_response)

    db_mock = AsyncMock()

    with patch("backend.services.rag_pipeline.enhance_query",
               new=AsyncMock(return_value="enhanced")), \
         patch("backend.services.rag_pipeline._generate_hyde_text",
               new=AsyncMock(return_value="hyde")), \
         patch("backend.services.rag_pipeline._generate_variants",
               new=AsyncMock(return_value=["v1", "v2"])), \
         patch("backend.services.rag_pipeline._encode_query",
               return_value=([0.1] * 768, None)), \
         patch("backend.services.rag_pipeline._pinecone_search",
               return_value=[{"metadata": {"child_id": child_id}}]), \
         patch("backend.services.rag_pipeline.rrf_merge",
               return_value=[child_id]), \
         patch("backend.services.rag_pipeline._fetch_child_chunks",
               new=AsyncMock(return_value=[fake_child])), \
         patch("backend.services.rag_pipeline.rerank_chunks",
               return_value=[fake_child]), \
         patch("backend.services.rag_pipeline._fetch_parent_chunks",
               new=AsyncMock(return_value=[fake_parent])), \
         patch("backend.services.rag_pipeline.compress_contexts",
               new=AsyncMock(return_value=[fake_compressed])), \
         patch("backend.services.rag_pipeline._RELEVANCE_PROMPT") as mock_rel_prompt, \
         patch("backend.services.rag_pipeline._HALLUCINATION_PROMPT") as mock_hal_prompt, \
         patch("backend.services.rag_pipeline.generate",
               new=AsyncMock(return_value={"answer": "answer", "sources": []})):

        mock_rel_prompt.__or__ = MagicMock(return_value=mock_rel_chain)
        mock_hal_prompt.__or__ = MagicMock(return_value=mock_hal_chain)

        pipeline = build_pipeline(db_mock)
        result = await pipeline.ainvoke(_make_initial_state())

    # CRAG retried once, then RELEVANT on second pass
    assert relevance_call_count == 2
    assert result["retry_count"] == 1
    assert "Refining search" in " ".join(result["status_events"])


# ── Hallucination retry test ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_hallucination_retry_uses_strict_prompt() -> None:
    """
    When check_hallucination returns 'ungrounded', the pipeline regenerates
    once with strict=True.  On the second check, it accepts whatever verdict.
    """
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    fake_child = _child_chunk(child_id, parent_id)
    fake_parent = _parent_chunk(parent_id)
    fake_compressed = {**fake_parent, "content": "Compressed."}

    relevance_response = MagicMock()
    relevance_response.content = "RELEVANT"
    # First hallucination check: ungrounded; second: grounded
    hal_responses = [MagicMock(), MagicMock()]
    hal_responses[0].content = "ungrounded"
    hal_responses[1].content = "grounded"

    mock_rel_chain = MagicMock()
    mock_rel_chain.ainvoke = AsyncMock(return_value=relevance_response)

    hal_call_count = 0

    async def hal_ainvoke(inputs: dict) -> MagicMock:
        nonlocal hal_call_count
        resp = hal_responses[min(hal_call_count, 1)]
        hal_call_count += 1
        return resp

    mock_hal_chain = MagicMock()
    mock_hal_chain.ainvoke = hal_ainvoke

    generate_calls: list[dict] = []

    async def mock_generate(query: str, chunks: list, strict: bool = False) -> dict:
        generate_calls.append({"strict": strict})
        return {"answer": "answer", "sources": []}

    db_mock = AsyncMock()

    with patch("backend.services.rag_pipeline.enhance_query",
               new=AsyncMock(return_value="enhanced")), \
         patch("backend.services.rag_pipeline._generate_hyde_text",
               new=AsyncMock(return_value="hyde")), \
         patch("backend.services.rag_pipeline._generate_variants",
               new=AsyncMock(return_value=["v1", "v2"])), \
         patch("backend.services.rag_pipeline._encode_query",
               return_value=([0.1] * 768, None)), \
         patch("backend.services.rag_pipeline._pinecone_search",
               return_value=[{"metadata": {"child_id": child_id}}]), \
         patch("backend.services.rag_pipeline.rrf_merge",
               return_value=[child_id]), \
         patch("backend.services.rag_pipeline._fetch_child_chunks",
               new=AsyncMock(return_value=[fake_child])), \
         patch("backend.services.rag_pipeline.rerank_chunks",
               return_value=[fake_child]), \
         patch("backend.services.rag_pipeline._fetch_parent_chunks",
               new=AsyncMock(return_value=[fake_parent])), \
         patch("backend.services.rag_pipeline.compress_contexts",
               new=AsyncMock(return_value=[fake_compressed])), \
         patch("backend.services.rag_pipeline._RELEVANCE_PROMPT") as mock_rel_prompt, \
         patch("backend.services.rag_pipeline._HALLUCINATION_PROMPT") as mock_hal_prompt, \
         patch("backend.services.rag_pipeline.generate", side_effect=mock_generate):

        mock_rel_prompt.__or__ = MagicMock(return_value=mock_rel_chain)
        mock_hal_prompt.__or__ = MagicMock(return_value=mock_hal_chain)

        pipeline = build_pipeline(db_mock)
        result = await pipeline.ainvoke(_make_initial_state())

    # generate called twice: first normal, then strict retry
    assert len(generate_calls) == 2
    assert generate_calls[0]["strict"] is False
    assert generate_calls[1]["strict"] is True
    assert result["hallucination_retry"] is True


# ── Empty retrieval test ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_empty_retrieval_skips_to_generate() -> None:
    """
    If Pinecone returns no results (RRF returns []), check_relevance marks
    IRRELEVANT and after max_crag_retries the pipeline falls through to
    generate_answer with an empty context.
    """
    relevance_response = MagicMock()
    relevance_response.content = "IRRELEVANT"

    hallucination_response = MagicMock()
    hallucination_response.content = "grounded"

    mock_rel_chain = MagicMock()
    mock_rel_chain.ainvoke = AsyncMock(return_value=relevance_response)
    mock_hal_chain = MagicMock()
    mock_hal_chain.ainvoke = AsyncMock(return_value=hallucination_response)

    db_mock = AsyncMock()

    with patch("backend.services.rag_pipeline.enhance_query",
               new=AsyncMock(return_value="enhanced")), \
         patch("backend.services.rag_pipeline._generate_hyde_text",
               new=AsyncMock(return_value="hyde")), \
         patch("backend.services.rag_pipeline._generate_variants",
               new=AsyncMock(return_value=["v1", "v2"])), \
         patch("backend.services.rag_pipeline._encode_query",
               return_value=([0.1] * 768, None)), \
         patch("backend.services.rag_pipeline._pinecone_search",
               return_value=[]), \
         patch("backend.services.rag_pipeline.rrf_merge",
               return_value=[]), \
         patch("backend.services.rag_pipeline._fetch_child_chunks",
               new=AsyncMock(return_value=[])), \
         patch("backend.services.rag_pipeline.rerank_chunks",
               return_value=[]), \
         patch("backend.services.rag_pipeline._fetch_parent_chunks",
               new=AsyncMock(return_value=[])), \
         patch("backend.services.rag_pipeline.compress_contexts",
               new=AsyncMock(return_value=[])), \
         patch("backend.services.rag_pipeline._RELEVANCE_PROMPT") as mock_rel_prompt, \
         patch("backend.services.rag_pipeline._HALLUCINATION_PROMPT") as mock_hal_prompt, \
         patch("backend.services.rag_pipeline.generate",
               new=AsyncMock(return_value={
                   "answer": "I could not find relevant information.",
                   "sources": [],
               })):

        mock_rel_prompt.__or__ = MagicMock(return_value=mock_rel_chain)
        mock_hal_prompt.__or__ = MagicMock(return_value=mock_hal_chain)

        pipeline = build_pipeline(db_mock)
        result = await pipeline.ainvoke(_make_initial_state())

    assert result["sources"] == []
    assert "could not find" in result["answer"]
