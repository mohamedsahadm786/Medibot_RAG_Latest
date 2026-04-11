"""
MediBot v2 — Chat endpoint (Phase 6)

POST /api/chat — invokes the LangGraph RAG pipeline, persists both
conversation turns to chat_messages, returns a complete JSON response.

Phase 8 will convert this to SSE streaming.
"""

import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.database import ChatMessage
from backend.schemas.chat import ChatRequest, ChatResponse, SourceCitation
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
      1. Save user message to chat_messages
      2. Run LangGraph RAG pipeline (enhance → HyDE → retrieve → rerank →
         compress → CRAG check → generate → hallucination check)
      3. Save assistant message to chat_messages
      4. Return answer + source citations
    """
    message_id = str(uuid.uuid4())
    session_uuid = _parse_session_id(request.session_id)

    # ── 1. Persist user message ───────────────────────────────────────────────
    user_msg = ChatMessage(
        id=uuid.uuid4(),
        session_id=session_uuid,
        role="user",
        content=request.query,
    )
    db.add(user_msg)
    await db.commit()

    # ── 2. Run LangGraph pipeline ─────────────────────────────────────────────
    initial_state: GraphState = {
        "query": request.query,
        "session_id": request.session_id,
        "summaries": [],              # Phase 7 will populate from memory
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

    # ── 3. Persist assistant message ──────────────────────────────────────────
    assistant_msg = ChatMessage(
        id=uuid.UUID(message_id),
        session_id=session_uuid,
        role="assistant",
        content=answer,
    )
    db.add(assistant_msg)
    await db.commit()

    # ── 4. Build and return response ──────────────────────────────────────────
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


def _parse_session_id(session_id: str) -> uuid.UUID:
    """Parse session_id string to UUID, generating a new one if invalid."""
    try:
        return uuid.UUID(session_id)
    except ValueError:
        return uuid.uuid4()
