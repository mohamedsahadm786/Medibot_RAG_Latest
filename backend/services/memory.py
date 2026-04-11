"""
MediBot v2 — Conversation Memory Service (Phase 7)

Sliding-window memory backed by the chat_messages table.

Each assistant row has a ``summary`` column that stores a single sentence
describing what was asked and answered in that turn.  Before each new query,
the last ``settings.memory_window`` summaries are fetched and passed to the
enhance_query node so it can resolve conversational references.

After each answer is generated, a new summary is produced via GPT-4o-mini
and written back to the assistant row.
"""

import logging
import uuid

import sqlalchemy as sa
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.models.database import ChatMessage

logger = logging.getLogger(__name__)

# ── Summary generation prompt ──────────────────────────────────────────────────

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical conversation summariser. Given one Q&A exchange, "
            "write a single concise sentence (max 80 words) summarising what the "
            "user asked and what the key answer was.\n"
            "Format: 'User asked about [topic]; assistant explained [key point(s)].'\n"
            "Return ONLY the summary sentence — no extra text.",
        ),
        (
            "human",
            "User question: {query}\n\nAssistant answer: {answer}\n\nSummary:",
        ),
    ]
)

_summary_llm: ChatOpenAI | None = None


def _get_summary_llm() -> ChatOpenAI:
    """Return a cached GPT-4o-mini instance for summary generation."""
    global _summary_llm
    if _summary_llm is None:
        _summary_llm = ChatOpenAI(
            model=settings.openai_model_aux,
            temperature=0.0,
            max_tokens=120,
            api_key=settings.openai_api_key,
        )
    return _summary_llm


# ── Public API ─────────────────────────────────────────────────────────────────

async def load_summaries(
    session_uuid: uuid.UUID,
    db: AsyncSession,
) -> list[str]:
    """
    Fetch the most recent conversation turn summaries for a session.

    Returns the last ``settings.memory_window`` non-null assistant summaries,
    sorted oldest-first so they read chronologically when passed to the LLM.

    Args:
        session_uuid:  Parsed UUID of the browser session.
        db:            Async SQLAlchemy session.

    Returns:
        List of summary strings (may be empty for first turn).
    """
    result = await db.execute(
        sa.select(ChatMessage.summary)
        .where(
            ChatMessage.session_id == session_uuid,
            ChatMessage.role == "assistant",
            ChatMessage.summary.is_not(None),
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(settings.memory_window)
    )
    rows: list[str] = result.scalars().all()
    # Reverse so summaries are in chronological order (oldest first)
    summaries = list(reversed(rows))
    logger.info(
        "Loaded %d conversation summaries for session %s",
        len(summaries),
        str(session_uuid)[:8],
    )
    return summaries


async def save_turn_summary(
    assistant_message_id: uuid.UUID,
    query: str,
    answer: str,
    db: AsyncSession,
) -> None:
    """
    Generate a 1-sentence summary of the current Q&A turn and persist it.

    The summary is written to the ``summary`` column of the assistant
    ``chat_messages`` row identified by ``assistant_message_id``.

    Args:
        assistant_message_id:  UUID of the already-persisted assistant row.
        query:                 The user's original raw query.
        answer:                The final generated answer.
        db:                    Async SQLAlchemy session.
    """
    summary = await _generate_summary(query, answer)
    await db.execute(
        sa.update(ChatMessage)
        .where(ChatMessage.id == assistant_message_id)
        .values(summary=summary)
    )
    await db.commit()
    logger.info(
        "Saved turn summary for message %s: %r",
        str(assistant_message_id)[:8],
        summary[:60],
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

async def _generate_summary(query: str, answer: str) -> str:
    """
    Call GPT-4o-mini to produce a single-sentence turn summary.

    Falls back to a truncated version of the query on failure.
    """
    try:
        llm = _get_summary_llm()
        chain = _SUMMARY_PROMPT | llm
        response = await chain.ainvoke({"query": query, "answer": answer[:800]})
        summary = response.content.strip()
        return summary if summary else f"User asked: {query[:120]}"
    except Exception as exc:
        logger.warning("Summary generation failed: %s", exc)
        return f"User asked: {query[:120]}"
