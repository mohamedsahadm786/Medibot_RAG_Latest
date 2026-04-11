"""
MediBot v2 — Chat endpoint (Phase 3)

POST /api/chat — accepts a query, runs hybrid retrieval + GPT-4o generation,
persists both turns to chat_messages, returns a complete JSON response.

Phase 8 will convert this to SSE streaming.
"""

import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.database import ChatMessage
from backend.schemas.chat import ChatRequest, ChatResponse, SourceCitation
from backend.services.retriever import retrieve
from backend.services.generator import generate

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
      2. Hybrid retrieval (Pinecone dense + sparse → parent chunks from PG)
      3. GPT-4o generation with medical prompt
      4. Save assistant message to chat_messages
      5. Return answer + source citations
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

    # ── 2. Retrieve relevant parent chunks ────────────────────────────────────
    try:
        chunks = await retrieve(query=request.query, db=db)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Retrieval service error.")

    # ── 3. Generate answer ────────────────────────────────────────────────────
    try:
        result = await generate(query=request.query, chunks=chunks)
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Generation service error.")

    answer: str = result["answer"]
    raw_sources: list[dict] = result["sources"]

    # ── 4. Persist assistant message ──────────────────────────────────────────
    assistant_msg = ChatMessage(
        id=uuid.UUID(message_id),
        session_id=session_uuid,
        role="assistant",
        content=answer,
    )
    db.add(assistant_msg)
    await db.commit()

    # ── 5. Build and return response ──────────────────────────────────────────
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
