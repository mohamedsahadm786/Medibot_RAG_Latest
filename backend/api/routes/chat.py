"""
MediBot v2 — Chat endpoints (Phase 8)

POST /api/chat        — JSON response (kept for testing / non-streaming clients)
POST /api/chat/stream — SSE stream: status events → tokens → sources → done
"""

import asyncio
import json
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from backend.database import get_db
from backend.models.database import ChatMessage
from backend.schemas.chat import ChatRequest, ChatResponse, SourceCitation
from backend.services.memory import load_summaries, save_turn_summary
from backend.services.rag_pipeline import GraphState, build_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Medical Q&A endpoint.

    Flow:
      1. Load last N conversation summaries (sliding-window memory)
      2. Save user message to chat_messages
      3. Run LangGraph RAG pipeline with summaries as context
      4. Save assistant message to chat_messages
      5. Generate + save 1-sentence turn summary for future context
      6. Return answer + source citations
    """
    message_id = str(uuid.uuid4())
    session_uuid = _parse_session_id(request.session_id)

    # ── 1. Load conversation memory ───────────────────────────────────────────
    summaries = await load_summaries(session_uuid, db)

    # ── 2. Persist user message ───────────────────────────────────────────────
    user_msg = ChatMessage(
        id=uuid.uuid4(),
        session_id=session_uuid,
        role="user",
        content=request.query,
    )
    db.add(user_msg)
    await db.commit()

    # ── 3. Run LangGraph pipeline ─────────────────────────────────────────────
    initial_state: GraphState = {
        "query": request.query,
        "session_id": request.session_id,
        "summaries": summaries,
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

    try:
        pipeline = build_pipeline(db)
        final_state: GraphState = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error("RAG pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="RAG pipeline error.")

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
            query=request.query,
            answer=answer,
            db=db,
        )
    except Exception as exc:
        # Non-critical: log and continue — missing summary only affects
        # future context, not the current response
        logger.warning("Failed to save turn summary: %s", exc)

    # ── 6. Build and return response ──────────────────────────────────────────
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
        session_id=request.session_id,
        answer=answer,
        sources=sources,
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> EventSourceResponse:
    """
    Streaming medical Q&A endpoint (SSE).

    SSE event sequence:
      {"type": "status", "step": "<step>", "message": "<text>"}  × N
      {"type": "token",  "content": "<word> "}                   × M
      {"type": "sources","data": [...]}
      {"type": "done",   "message_id": "<uuid>", "session_id": "<sid>"}

    On pipeline failure:
      {"type": "error", "message": "<reason>"}
    """
    message_id = str(uuid.uuid4())
    session_uuid = _parse_session_id(request.session_id)

    # ── 1. Load conversation memory ───────────────────────────────────────────
    summaries = await load_summaries(session_uuid, db)

    # ── 2. Persist user message ───────────────────────────────────────────────
    user_msg = ChatMessage(
        id=uuid.uuid4(),
        session_id=session_uuid,
        role="user",
        content=request.query,
    )
    db.add(user_msg)
    await db.commit()

    # ── 3. Build initial pipeline state ──────────────────────────────────────
    initial_state: GraphState = {
        "query": request.query,
        "session_id": request.session_id,
        "summaries": summaries,
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

    async def run_pipeline() -> None:
        """Run the LangGraph pipeline and push results to the event queue."""
        try:
            pipeline = build_pipeline(db, event_queue=event_queue)
            final_state: GraphState = await pipeline.ainvoke(initial_state)
            await event_queue.put({"type": "final", "state": final_state})
        except Exception as exc:
            logger.error("SSE pipeline error: %s", exc, exc_info=True)
            await event_queue.put({"type": "error", "message": str(exc)})
        finally:
            await event_queue.put(None)  # sentinel: generator should stop

    async def generate():
        """Yield SSE-formatted strings for the EventSourceResponse."""
        pipeline_task = asyncio.create_task(run_pipeline())
        final_state: GraphState | None = None

        try:
            # ── Drain status events while pipeline runs ────────────────────
            while True:
                event = await event_queue.get()
                if event is None:
                    break

                event_type = event.get("type")

                if event_type == "status":
                    yield json.dumps(event)
                elif event_type == "final":
                    final_state = event["state"]
                elif event_type == "error":
                    yield json.dumps({"type": "error", "message": event["message"]})
                    return

            if final_state is None:
                yield json.dumps({"type": "error", "message": "Pipeline returned no state."})
                return

            answer: str = final_state["answer"]
            raw_sources: list[dict] = final_state["sources"]

            # ── Stream answer word by word ─────────────────────────────────
            if answer:
                words = answer.split(" ")
                for i, word in enumerate(words):
                    token = word if i == len(words) - 1 else word + " "
                    yield json.dumps({"type": "token", "content": token})

            # ── Persist assistant message ──────────────────────────────────
            assistant_msg = ChatMessage(
                id=uuid.UUID(message_id),
                session_id=session_uuid,
                role="assistant",
                content=answer,
            )
            db.add(assistant_msg)
            await db.commit()

            # ── Save turn summary (non-critical) ───────────────────────────
            try:
                await save_turn_summary(
                    assistant_message_id=uuid.UUID(message_id),
                    query=request.query,
                    answer=answer,
                    db=db,
                )
            except Exception as exc:
                logger.warning("Failed to save turn summary (stream): %s", exc)

            # ── Emit sources ───────────────────────────────────────────────
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

            # ── Done ───────────────────────────────────────────────────────
            yield json.dumps({
                "type": "done",
                "message_id": message_id,
                "session_id": request.session_id,
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
