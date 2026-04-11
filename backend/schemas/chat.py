from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    session_id: str = Field(
        ...,
        description="Browser-generated UUID for session isolation. No auth required.",
        min_length=1,
        max_length=36,
    )
    query: str = Field(
        ...,
        description="User's medical question.",
        min_length=1,
        max_length=2000,
    )


class SourceCitation(BaseModel):
    """A single source citation attached to an answer."""

    chunk_id: str
    page_number: int
    section_heading: str
    excerpt: str
    source_pdf: str


class ChatResponse(BaseModel):
    """Complete non-streaming chat response (used in Phase 3, replaced by SSE in Phase 8)."""

    message_id: str
    session_id: str
    answer: str
    sources: list[SourceCitation] = []


class SSEEvent(BaseModel):
    """Server-Sent Event payload — used from Phase 8 onward."""

    type: str  # status | token | citation | done | error
    content: str | list | None = None
    message_id: str | None = None
