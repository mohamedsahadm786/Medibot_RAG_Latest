"""
MediBot v2 — LangGraph Agentic RAG Pipeline (Phase 6)

Wraps the entire retrieval → reranking → compression → generation flow in a
LangGraph StateGraph with two conditional feedback loops:

  1. CRAG (Corrective RAG) — after relevance check, retry once with a
     modified query if retrieved context is irrelevant.
  2. Hallucination guard — after generation, regenerate with a stricter
     prompt if any claim is ungrounded (max one retry).

Graph topology:
  enhance_query
      → hyde_and_multiquery
          → retrieve
              → rerank
                  → fetch_and_compress
                      → check_relevance
                          ├─ RELEVANT/PARTIAL → generate_answer
                          └─ IRRELEVANT (retry_count < max) → _prepare_crag_retry
                                                                    → enhance_query
                          └─ IRRELEVANT (retry_count >= max) → generate_answer
                      generate_answer
                          → check_hallucination
                              ├─ grounded → END
                              └─ ungrounded (first) → _prepare_hallucination_retry
                                                           → generate_answer
                              └─ ungrounded (second) → END
"""

import asyncio
import logging
import uuid
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import TypedDict

from backend.config import settings
from backend.services.compressor import compress_contexts
from backend.services.generator import generate
from backend.services.query_transform import (
    _generate_hyde_text,
    _generate_variants,
    enhance_query,
)
from backend.services.reranker import rerank_chunks, rrf_merge
from backend.services.retriever import (
    _encode_query,
    _fetch_child_chunks,
    _fetch_parent_chunks,
    _pinecone_search,
)

logger = logging.getLogger(__name__)

# ── State schema ───────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """Full state carried through the LangGraph pipeline."""

    # ── Input ──────────────────────────────────────────────────────────────────
    query: str
    session_id: str
    summaries: list[str]

    # ── Pipeline state ─────────────────────────────────────────────────────────
    enhanced_query: str
    hyde_answer: str
    query_variants: list[str]
    retrieved_chunks: list[dict]       # RRF-ranked [{child_id: str}]
    reranked_chunks: list[dict]        # top-5 child chunk dicts (with content)
    parent_chunks: list[dict]          # fetched parent chunk dicts
    compressed_contexts: list[dict]    # after contextual compression

    # ── Verdict fields ─────────────────────────────────────────────────────────
    relevance_verdict: str             # RELEVANT | PARTIAL | IRRELEVANT
    answer: str
    sources: list[dict]
    hallucination_verdict: str         # grounded | ungrounded

    # ── Control flow ───────────────────────────────────────────────────────────
    retry_count: int                   # CRAG retry counter (starts at 0)
    hallucination_retry: bool          # True after first hallucination retry
    status_events: list[str]           # human-readable progress events


# ── Prompts for CRAG and hallucination checks ──────────────────────────────────

_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical relevance judge. Given a query and retrieved context, "
            "assess whether the context is sufficient to answer the query.\n\n"
            "Return ONLY one of these three words:\n"
            "  RELEVANT   — context directly answers the query\n"
            "  PARTIAL    — context is related but incomplete\n"
            "  IRRELEVANT — context does not address the query",
        ),
        (
            "human",
            "Query: {query}\n\nContext:\n{context}\n\nVerdict:",
        ),
    ]
)

_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical fact-checker. Verify whether every factual claim in "
            "the given answer is explicitly supported by the provided context passages.\n\n"
            "Return ONLY one of these two words:\n"
            "  grounded   — all claims are supported by the context\n"
            "  ungrounded — at least one claim is not found in the context",
        ),
        (
            "human",
            "Context:\n{context}\n\nAnswer:\n{answer}\n\nVerdict:",
        ),
    ]
)

_aux_llm: ChatOpenAI | None = None


def _get_aux_llm() -> ChatOpenAI:
    """Return a cached GPT-4o-mini instance for verdict calls."""
    global _aux_llm
    if _aux_llm is None:
        _aux_llm = ChatOpenAI(
            model=settings.openai_model_aux,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )
    return _aux_llm


# ── Context formatter (shared between relevance + hallucination checks) ────────

def _format_context(chunks: list[dict]) -> str:
    """Format compressed chunks into a readable context block."""
    return "\n\n---\n\n".join(
        f"[Page {c.get('page_number', '?')}] {c.get('content', '')}"
        for c in chunks
    )


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_pipeline(
    db: AsyncSession,
    event_queue: asyncio.Queue | None = None,
):
    """
    Build and compile the LangGraph RAG pipeline.

    The database session is captured via closure so it doesn't need to
    live in the graph state (it's not JSON-serialisable).

    Args:
        db:          Open async SQLAlchemy session for PostgreSQL queries.
        event_queue: Optional asyncio.Queue for SSE streaming. When provided,
                     nodes push ``{"type": "status", "step": ..., "message": ...}``
                     dicts so the SSE endpoint can forward them in real time.

    Returns:
        A compiled LangGraph graph ready for ``ainvoke()``.
    """

    # ── Node 1: enhance_query ──────────────────────────────────────────────────

    async def node_enhance_query(state: GraphState) -> dict:
        """Rewrite the raw query using conversation context."""
        is_crag_retry = state["retry_count"] > 0
        events = state.get("status_events", [])

        if is_crag_retry:
            events = events + ["Refining search for better results..."]
            if event_queue is not None:
                await event_queue.put({
                    "type": "status",
                    "step": "refining",
                    "message": "Refining search for better results...",
                })
        else:
            if event_queue is not None:
                await event_queue.put({
                    "type": "status",
                    "step": "thinking",
                    "message": "Understanding your question...",
                })

        enhanced = await enhance_query(state["query"], state["summaries"])
        logger.info("enhance_query → %r", enhanced[:80])
        return {
            "enhanced_query": enhanced,
            "status_events": events,
        }

    # ── Node 2: hyde_and_multiquery ────────────────────────────────────────────

    async def node_hyde_and_multiquery(state: GraphState) -> dict:
        """Generate HyDE hypothetical answer and 2 query variants concurrently."""
        hyde_text, variants = await asyncio.gather(
            _generate_hyde_text(state["enhanced_query"]),
            _generate_variants(state["enhanced_query"]),
        )
        logger.info(
            "hyde_and_multiquery → HyDE %d chars, %d variants",
            len(hyde_text),
            len(variants),
        )
        return {"hyde_answer": hyde_text, "query_variants": variants}

    # ── Node 3: retrieve ───────────────────────────────────────────────────────

    async def node_retrieve(state: GraphState) -> dict:
        """Encode all 3 texts, run Pinecone hybrid search, RRF merge."""
        hyde_dense, hyde_sparse = _encode_query(state["hyde_answer"])
        var1_dense, var1_sparse = _encode_query(state["query_variants"][0])
        var2_dense, var2_sparse = _encode_query(state["query_variants"][1])

        hyde_matches = _pinecone_search(hyde_dense, hyde_sparse, top_k=10)
        var1_matches = _pinecone_search(var1_dense, var1_sparse, top_k=5)
        var2_matches = _pinecone_search(var2_dense, var2_sparse, top_k=5)

        logger.info(
            "retrieve — Pinecone matches: HyDE=%d var1=%d var2=%d",
            len(hyde_matches), len(var1_matches), len(var2_matches),
        )

        ranked_ids = rrf_merge([hyde_matches, var1_matches, var2_matches])
        retrieved = [{"child_id": cid} for cid in ranked_ids]

        if event_queue is not None:
            await event_queue.put({
                "type": "status",
                "step": "searching",
                "message": "Searching medical knowledge base...",
            })

        events = state.get("status_events", []) + ["Searching medical knowledge base..."]
        return {
            "retrieved_chunks": retrieved,
            "status_events": events,
        }

    # ── Node 4: rerank ─────────────────────────────────────────────────────────

    async def node_rerank(state: GraphState) -> dict:
        """Fetch child chunk text from PostgreSQL and cross-encoder rerank."""
        child_ids: list[uuid.UUID] = []
        for item in state["retrieved_chunks"]:
            try:
                child_ids.append(uuid.UUID(item["child_id"]))
            except (ValueError, KeyError):
                pass

        child_chunks = await _fetch_child_chunks(child_ids, db)
        top_children = rerank_chunks(state["enhanced_query"], child_chunks, top_n=5)

        if event_queue is not None:
            await event_queue.put({
                "type": "status",
                "step": "analyzing",
                "message": "Analyzing relevant sources...",
            })

        events = state.get("status_events", []) + ["Analyzing relevant sources..."]
        return {
            "reranked_chunks": top_children,
            "status_events": events,
        }

    # ── Node 5: fetch_and_compress ─────────────────────────────────────────────

    async def node_fetch_and_compress(state: GraphState) -> dict:
        """Map top child chunks to unique parent IDs, fetch parents, compress."""
        parent_ids: list[uuid.UUID] = []
        seen: set[str] = set()
        for chunk in state["reranked_chunks"]:
            pid_str = chunk.get("parent_id")
            if pid_str and pid_str not in seen:
                seen.add(pid_str)
                try:
                    parent_ids.append(uuid.UUID(pid_str))
                except ValueError:
                    logger.warning("Invalid parent_id: %s", pid_str)

        parent_chunks = await _fetch_parent_chunks(parent_ids, db)
        compressed = await compress_contexts(state["enhanced_query"], parent_chunks)

        logger.info(
            "fetch_and_compress — %d parent(s), %d compressed",
            len(parent_chunks),
            len(compressed),
        )
        return {
            "parent_chunks": parent_chunks,
            "compressed_contexts": compressed,
        }

    # ── Node 6: check_relevance ────────────────────────────────────────────────

    async def node_check_relevance(state: GraphState) -> dict:
        """CRAG: judge whether retrieved context is sufficient."""
        if not state["compressed_contexts"]:
            logger.info("check_relevance — no context, marking IRRELEVANT")
            return {
                "relevance_verdict": "IRRELEVANT",
                "retry_count": state["retry_count"] + 1,
            }

        context = _format_context(state["compressed_contexts"])
        llm = _get_aux_llm()
        chain = _RELEVANCE_PROMPT | llm
        response = await chain.ainvoke(
            {"query": state["enhanced_query"], "context": context}
        )
        verdict = response.content.strip().upper()
        if verdict not in {"RELEVANT", "PARTIAL", "IRRELEVANT"}:
            verdict = "PARTIAL"  # safe fallback

        new_retry = (
            state["retry_count"] + 1
            if verdict == "IRRELEVANT"
            else state["retry_count"]
        )

        logger.info(
            "check_relevance → %s (retry_count was %d)",
            verdict,
            state["retry_count"],
        )
        return {"relevance_verdict": verdict, "retry_count": new_retry}

    # ── Node 7: generate_answer ────────────────────────────────────────────────

    async def node_generate_answer(state: GraphState) -> dict:
        """GPT-4o answer generation. Uses stricter prompt on hallucination retry."""
        chunks = state["compressed_contexts"]
        verdict = state.get("relevance_verdict", "RELEVANT")
        strict = state.get("hallucination_retry", False)

        # Prepend instruction note for edge cases
        query = state["enhanced_query"]
        if verdict == "IRRELEVANT":
            query = (
                f"{query}\n\n[Note: retrieved context may be insufficient. "
                "Answer only with what is available, and clearly state limitations.]"
            )
        elif verdict == "PARTIAL":
            query = (
                f"{query}\n\n[Note: context is partially relevant. "
                "Clearly note any gaps in the available information.]"
            )

        if event_queue is not None:
            await event_queue.put({
                "type": "status",
                "step": "generating",
                "message": "",
            })

        result = await generate(query=query, chunks=chunks, strict=strict)
        logger.info(
            "generate_answer → %d chars, %d sources (strict=%s)",
            len(result["answer"]),
            len(result["sources"]),
            strict,
        )
        return {"answer": result["answer"], "sources": result["sources"]}

    # ── Node 8: check_hallucination ────────────────────────────────────────────

    async def node_check_hallucination(state: GraphState) -> dict:
        """Verify all answer claims are grounded in the context."""
        if not state["compressed_contexts"]:
            # No context was used → trivially grounded (it said "no info found")
            return {"hallucination_verdict": "grounded"}

        context = _format_context(state["compressed_contexts"])
        llm = _get_aux_llm()
        chain = _HALLUCINATION_PROMPT | llm
        response = await chain.ainvoke(
            {"context": context, "answer": state["answer"]}
        )
        verdict = response.content.strip().lower()
        if verdict not in {"grounded", "ungrounded"}:
            verdict = "grounded"  # safe fallback

        logger.info("check_hallucination → %s", verdict)
        return {"hallucination_verdict": verdict}

    # ── Transition nodes (state mutation before re-routing) ────────────────────

    async def node_prepare_crag_retry(state: GraphState) -> dict:
        """Increment retry counter before looping back to enhance_query."""
        # retry_count was already incremented by check_relevance; nothing extra needed.
        # This node exists to give the graph a named hop for clarity.
        return {}

    async def node_prepare_hallucination_retry(state: GraphState) -> dict:
        """Mark hallucination retry so generate_answer uses the strict prompt."""
        return {"hallucination_retry": True}

    # ── Conditional edge functions ─────────────────────────────────────────────

    def route_after_relevance(state: GraphState) -> str:
        verdict = state.get("relevance_verdict", "RELEVANT")
        retry_count = state.get("retry_count", 0)

        if verdict == "IRRELEVANT" and retry_count <= settings.max_crag_retries:
            logger.info("route_after_relevance → crag_retry (count=%d)", retry_count)
            return "crag_retry"
        logger.info("route_after_relevance → generate_answer (verdict=%s)", verdict)
        return "generate_answer"

    def route_after_hallucination(state: GraphState) -> str:
        verdict = state.get("hallucination_verdict", "grounded")
        already_retried = state.get("hallucination_retry", False)

        if verdict == "grounded" or already_retried:
            logger.info("route_after_hallucination → END")
            return END
        logger.info("route_after_hallucination → hallucination_retry")
        return "hallucination_retry"

    # ── Build graph ────────────────────────────────────────────────────────────

    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("enhance_query",             node_enhance_query)
    graph.add_node("hyde_and_multiquery",        node_hyde_and_multiquery)
    graph.add_node("retrieve",                   node_retrieve)
    graph.add_node("rerank",                     node_rerank)
    graph.add_node("fetch_and_compress",         node_fetch_and_compress)
    graph.add_node("check_relevance",            node_check_relevance)
    graph.add_node("generate_answer",            node_generate_answer)
    graph.add_node("check_hallucination",        node_check_hallucination)
    graph.add_node("crag_retry",                 node_prepare_crag_retry)
    graph.add_node("hallucination_retry",        node_prepare_hallucination_retry)

    # Entry point
    graph.set_entry_point("enhance_query")

    # Linear edges
    graph.add_edge("enhance_query",         "hyde_and_multiquery")
    graph.add_edge("hyde_and_multiquery",   "retrieve")
    graph.add_edge("retrieve",              "rerank")
    graph.add_edge("rerank",                "fetch_and_compress")
    graph.add_edge("fetch_and_compress",    "check_relevance")

    # Conditional edge: after relevance check
    graph.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {"crag_retry": "crag_retry", "generate_answer": "generate_answer"},
    )

    # CRAG retry loop
    graph.add_edge("crag_retry", "enhance_query")

    # Linear: generate → check hallucination
    graph.add_edge("generate_answer", "check_hallucination")

    # Conditional edge: after hallucination check
    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {END: END, "hallucination_retry": "hallucination_retry"},
    )

    # Hallucination retry → regenerate
    graph.add_edge("hallucination_retry", "generate_answer")

    return graph.compile()
