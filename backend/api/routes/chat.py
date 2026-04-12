"""
MediBot v2 — Chat endpoints (Phase 11)

POST /api/chat        — JSON response (kept for testing / non-streaming clients)
POST /api/chat/stream — SSE stream: status events → tokens → sources → done

Both endpoints check the semantic cache before running the pipeline.
Cache hits skip the pipeline and return (or stream) the stored answer.
Prometheus metrics (cache hits/misses) are incremented on every request.
"""

import asyncio
import json
import time
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from backend.core.metrics import cache_hits_total, cache_misses_total
from backend.database import get_db
from backend.models.database import ChatMessage
from backend.schemas.chat import ChatRequest, ChatResponse, SourceCitation
from backend.services.cache import check_cache, store_cache
from backend.services.evaluation import evaluate_with_ragas, log_retrieval_details
from backend.services.memory import load_summaries, save_turn_summary
from backend.services.rag_pipeline import GraphState, build_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{10}/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Medical Q&A endpoint (non-streaming).

    Flow:
      0. Semantic cache check — return cached answer if similarity ≥ 0.95
      1. Load last N conversation summaries (sliding-window memory)
      2. Save user message to chat_messages
      3. Run LangGraph RAG pipeline with summaries as context
      4. Save assistant message to chat_messages
      5. Generate + save 1-sentence turn summary for future context
      6. Store result in semantic cache
      7. Return answer + source citations
    """
    # ── 0. Semantic cache check ───────────────────────────────────────────────
    try:
        cached = await check_cache(body.query)
    except Exception as exc:
        logger.warning("Cache check failed: %s", exc)
        cached = None

    if cached is not None:
        cache_hits_total.inc()
        return ChatResponse(
            message_id=str(uuid.uuid4()),
            session_id=body.session_id,
            answer=cached["answer"],
            sources=[SourceCitation(**s) for s in cached["sources"]],
        )

    cache_misses_total.inc()

    # ── 1. Load conversation memory ───────────────────────────────────────────
    message_id = str(uuid.uuid4())
    session_uuid = _parse_session_id(body.session_id)

    summaries = await load_summaries(session_uuid, db)

    # ── 2. Persist user message ───────────────────────────────────────────────
    user_msg = ChatMessage(
        id=uuid.uuid4(),
        session_id=session_uuid,
        role="user",
        content=body.query,
    )
    db.add(user_msg)
    await db.commit()

    # ── 3. Run LangGraph pipeline ─────────────────────────────────────────────
    initial_state: GraphState = {
        "query": body.query,
        "session_id": body.session_id,
        "summaries": summaries,
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

    _pipeline_start = time.perf_counter()
    try:
        pipeline = build_pipeline(db)
        final_state: GraphState = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error("RAG pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="RAG pipeline error.")
    _pipeline_latency_ms = int((time.perf_counter() - _pipeline_start) * 1000)

    answer: str = final_state["answer"]
    raw_sources: list[dict] = final_state["sources"]

    # ── 4. Persist assistant message ──────────────────────────────────────────
    assistant_msg = ChatMessage(
        id=uuid.UUID(message_id),
        session_id=session_uuid,
        role="assistant",
        content=answer,
    )
    db.add(assistant_msg)
    await db.commit()

    # ── 5. Generate and persist turn summary ──────────────────────────────────
    try:
        await save_turn_summary(
            assistant_message_id=uuid.UUID(message_id),
            query=body.query,
            answer=answer,
            db=db,
        )
    except Exception as exc:
        logger.warning("Failed to save turn summary: %s", exc)

    # ── 6. Store result in semantic cache ─────────────────────────────────────
    try:
        await store_cache(body.query, answer, raw_sources)
    except Exception as exc:
        logger.warning("Failed to store cache: %s", exc)

    # ── 7. Queue background evaluation tasks (medical queries only) ───────────
    # Skip RAGAS evaluation for conversational queries — they have no RAG
    # context and would produce meaningless 0% scores that pollute averages.
    is_medical = final_state.get("intent", "medical") != "conversational"
    if is_medical:
        try:
            contexts = [c.get("content", "") for c in final_state["compressed_contexts"]]
            log_retrieval_details.delay(
                message_id=message_id,
                enhanced_query=final_state["enhanced_query"],
                hyde_answer=final_state["hyde_answer"],
                retrieved_child_ids=[c["child_id"] for c in final_state["retrieved_chunks"]],
                retrieved_parent_ids=[c.get("chunk_id", "") for c in final_state["parent_chunks"]],
                reranker_scores={
                    c.get("chunk_id", f"chunk_{i}"): float(i + 1)
                    for i, c in enumerate(final_state["reranked_chunks"])
                },
                relevance_verdict=final_state["relevance_verdict"],
                hallucination_verdict=final_state["hallucination_verdict"],
                latency_ms=_pipeline_latency_ms,
            )
            evaluate_with_ragas.delay(
                message_id=message_id,
                query=body.query,
                answer=answer,
                contexts=contexts,
            )
        except Exception as exc:
            logger.warning("Failed to queue evaluation tasks: %s", exc)

    # ── 8. Build and return response ──────────────────────────────────────────
    sources = [
        SourceCitation(
            chunk_id=s["chunk_id"],
            page_number=s["page_number"],
            section_heading=s["section_heading"],
            excerpt=s["excerpt"],
            source_pdf=s["source_pdf"],
        )
        for s in raw_sources
    ]

    return ChatResponse(
        message_id=message_id,
        session_id=body.session_id,
        answer=answer,
        sources=sources,
    )


@router.post("/chat/stream")
@limiter.limit(f"{10}/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> EventSourceResponse:
    """
    Streaming medical Q&A endpoint (SSE).

    SSE event sequence (cache miss):
      {"type": "status", "step": "<step>", "message": "<text>"}  × N
      {"type": "token",  "content": "<word> "}                   × M
      {"type": "sources","data": [...]}
      {"type": "done",   "message_id": "<uuid>", "session_id": "<sid>"}

    SSE event sequence (cache hit):
      {"type": "token",  "content": "<word> "}                   × M
      {"type": "sources","data": [...]}
      {"type": "done",   "message_id": "<uuid>", "session_id": "<sid>"}

    On pipeline failure:
      {"type": "error", "message": "<reason>"}
    """
    message_id = str(uuid.uuid4())
    session_uuid = _parse_session_id(body.session_id)

    async def generate():
        """Yield SSE-formatted strings for the EventSourceResponse."""

        # ── 0. Semantic cache check ────────────────────────────────────────
        try:
            cached = await check_cache(body.query)
        except Exception as exc:
            logger.warning("Cache check failed (stream): %s", exc)
            cached = None

        if cached is not None:
            cache_hits_total.inc()
            cached_answer: str = cached.get("answer", "")
            if cached_answer:
                words = cached_answer.split(" ")
                for i, word in enumerate(words):
                    token = word if i == len(words) - 1 else word + " "
                    yield json.dumps({"type": "token", "content": token})
            yield json.dumps({"type": "sources", "data": cached.get("sources", [])})
            yield json.dumps({
                "type": "done",
                "message_id": message_id,
                "session_id": body.session_id,
            })
            return

        cache_misses_total.inc()

        # ── 1. Load conversation memory ────────────────────────────────────
        summaries = await load_summaries(session_uuid, db)

        # ── 2. Persist user message ────────────────────────────────────────
        user_msg = ChatMessage(
            id=uuid.uuid4(),
            session_id=session_uuid,
            role="user",
            content=body.query,
        )
        db.add(user_msg)
        await db.commit()

        # ── 3. Build initial pipeline state ───────────────────────────────
        initial_state: GraphState = {
            "query": body.query,
            "session_id": body.session_id,
            "summaries": summaries,
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

        event_queue: asyncio.Queue = asyncio.Queue()
        _pipeline_start = time.perf_counter()

        async def run_pipeline() -> None:
            try:
                pipeline = build_pipeline(db, event_queue=event_queue)
                final_state: GraphState = await pipeline.ainvoke(initial_state)
                latency_ms = int((time.perf_counter() - _pipeline_start) * 1000)
                await event_queue.put({"type": "final", "state": final_state, "latency_ms": latency_ms})
            except Exception as exc:
                logger.error("SSE pipeline error: %s", exc, exc_info=True)
                await event_queue.put({"type": "error", "message": str(exc)})
            finally:
                await event_queue.put(None)  # sentinel

        pipeline_task = asyncio.create_task(run_pipeline())
        final_state: GraphState | None = None

        try:
            # ── 4. Drain status events while pipeline runs ─────────────────
            while True:
                event = await event_queue.get()
                if event is None:
                    break

                event_type = event.get("type")

                if event_type == "status":
                    yield json.dumps(event)
                elif event_type == "final":
                    final_state = event["state"]
                    _stream_latency_ms: int = event.get("latency_ms", 0)
                elif event_type == "error":
                    yield json.dumps({"type": "error", "message": event["message"]})
                    return

            if final_state is None:
                yield json.dumps({"type": "error", "message": "Pipeline returned no state."})
                return

            answer: str = final_state["answer"]
            raw_sources: list[dict] = final_state["sources"]

            # ── 5. Stream answer word by word ──────────────────────────────
            if answer:
                words = answer.split(" ")
                for i, word in enumerate(words):
                    token = word if i == len(words) - 1 else word + " "
                    yield json.dumps({"type": "token", "content": token})

            # ── 6. Persist assistant message ───────────────────────────────
            assistant_msg = ChatMessage(
                id=uuid.UUID(message_id),
                session_id=session_uuid,
                role="assistant",
                content=answer,
            )
            db.add(assistant_msg)
            await db.commit()

            # ── 7. Save turn summary (non-critical) ────────────────────────
            try:
                await save_turn_summary(
                    assistant_message_id=uuid.UUID(message_id),
                    query=body.query,
                    answer=answer,
                    db=db,
                )
            except Exception as exc:
                logger.warning("Failed to save turn summary (stream): %s", exc)

            # ── 8. Store result in semantic cache (non-critical) ───────────
            try:
                await store_cache(body.query, answer, raw_sources)
            except Exception as exc:
                logger.warning("Failed to store cache (stream): %s", exc)

            # ── 9. Queue background evaluation tasks (medical queries only) ──
            is_medical = final_state.get("intent", "medical") != "conversational"
            if is_medical:
                try:
                    contexts = [c.get("content", "") for c in final_state["compressed_contexts"]]
                    log_retrieval_details.delay(
                        message_id=message_id,
                        enhanced_query=final_state["enhanced_query"],
                        hyde_answer=final_state["hyde_answer"],
                        retrieved_child_ids=[c["child_id"] for c in final_state["retrieved_chunks"]],
                        retrieved_parent_ids=[c.get("chunk_id", "") for c in final_state["parent_chunks"]],
                        reranker_scores={
                            c.get("chunk_id", f"chunk_{i}"): float(i + 1)
                            for i, c in enumerate(final_state["reranked_chunks"])
                        },
                        relevance_verdict=final_state["relevance_verdict"],
                        hallucination_verdict=final_state["hallucination_verdict"],
                        latency_ms=_stream_latency_ms,
                    )
                    evaluate_with_ragas.delay(
                        message_id=message_id,
                        query=body.query,
                        answer=answer,
                        contexts=contexts,
                    )
                except Exception as exc:
                    logger.warning("Failed to queue evaluation tasks (stream): %s", exc)

            # ── 10. Emit sources and done ──────────────────────────────────
            sources = [
                SourceCitation(
                    chunk_id=s["chunk_id"],
                    page_number=s["page_number"],
                    section_heading=s["section_heading"],
                    excerpt=s["excerpt"],
                    source_pdf=s["source_pdf"],
                ).model_dump()
                for s in raw_sources
            ]
            yield json.dumps({"type": "sources", "data": sources})
            yield json.dumps({
                "type": "done",
                "message_id": message_id,
                "session_id": body.session_id,
            })

        finally:
            if not pipeline_task.done():
                pipeline_task.cancel()

    return EventSourceResponse(generate())


def _parse_session_id(session_id: str) -> uuid.UUID:
    """Parse session_id string to UUID, generating a new one if invalid."""
    try:
        return uuid.UUID(session_id)
    except ValueError:
        return uuid.uuid4()
