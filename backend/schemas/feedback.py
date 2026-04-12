from typing import Literal

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Thumbs-up / thumbs-down feedback for an assistant message."""

    message_id: str = Field(..., description="UUID of the assistant ChatMessage.")
    feedback: Literal["up", "down"] = Field(..., description="'up' or 'down'.")


class FeedbackResponse(BaseModel):
    """Confirmation of recorded feedback."""

    feedback_id: str
    message_id: str
    feedback: str
