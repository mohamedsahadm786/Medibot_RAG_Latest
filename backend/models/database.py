import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class ChatMessage(Base):
    """Stores every user and assistant message with an optional 1-sentence summary."""

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    retrieval_log: Mapped[Optional["RetrievalLog"]] = relationship(
        back_populates="message", uselist=False
    )
    feedback: Mapped[Optional["UserFeedback"]] = relationship(
        back_populates="message", uselist=False
    )


class DocumentChunk(Base):
    """Stores parent and child chunks from the medical PDF."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chunk_type: Mapped[str] = mapped_column(String(10), nullable=False)  # parent | child
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id"),
        nullable=True,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_heading: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source_pdf: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )


class RetrievalLog(Base):
    """Full audit trail for each RAG pipeline execution."""

    __tablename__ = "retrieval_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_messages.id"), nullable=True
    )
    enhanced_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    hyde_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieved_child_ids: Mapped[Optional[list]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True
    )
    retrieved_parent_ids: Mapped[Optional[list]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True
    )
    reranker_scores: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    relevance_verdict: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    hallucination_verdict: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    ragas_scores: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    total_tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    message: Mapped[Optional["ChatMessage"]] = relationship(back_populates="retrieval_log")


class UserFeedback(Base):
    """Thumbs up/down feedback per assistant message."""

    __tablename__ = "user_feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_messages.id"), nullable=True
    )
    feedback: Mapped[str] = mapped_column(String(4), nullable=False)  # up | down
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    message: Mapped[Optional["ChatMessage"]] = relationship(back_populates="feedback")
