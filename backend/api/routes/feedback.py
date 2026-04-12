"""
MediBot v2 — Feedback endpoint (Phase 13)

POST /api/feedback — record thumbs-up or thumbs-down for an assistant message.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.metrics import user_feedback_total
from backend.database import get_db
from backend.models.database import ChatMessage, UserFeedback
from backend.schemas.feedback import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """
    Record thumbs-up or thumbs-down feedback for an assistant message.

    The message_id must correspond to an existing assistant ChatMessage.
    Submitting feedback twice for the same message replaces the previous vote.
    """
    try:
        msg_uuid = uuid.UUID(request.message_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid message_id format.") from exc

    # Verify the message exists
    msg_q = await db.execute(
        select(ChatMessage).where(
            ChatMessage.id == msg_uuid,
            ChatMessage.role == "assistant",
        )
    )
    if msg_q.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Message not found.")

    # Upsert: delete existing feedback for this message then insert new
    existing_q = await db.execute(
        select(UserFeedback).where(UserFeedback.message_id == msg_uuid)
    )
    existing = existing_q.scalar_one_or_none()
    if existing is not None:
        await db.delete(existing)

    feedback_row = UserFeedback(
        id=uuid.uuid4(),
        message_id=msg_uuid,
        feedback=request.feedback,
    )
    db.add(feedback_row)
    await db.commit()

    user_feedback_total.labels(feedback_type=request.feedback).inc()
    logger.info("Feedback '%s' recorded for message %s", request.feedback, msg_uuid)

    return FeedbackResponse(
        feedback_id=str(feedback_row.id),
        message_id=request.message_id,
        feedback=request.feedback,
    )
