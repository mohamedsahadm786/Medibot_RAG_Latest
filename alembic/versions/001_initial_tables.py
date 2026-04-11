"""Initial tables: chat_messages, document_chunks, retrieval_logs, user_feedback

Revision ID: 001
Revises:
Create Date: 2026-04-11

"""
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # document_chunks
    # Must be created before chat_messages (no FK dependency),
    # and before retrieval_logs (no direct FK here).
    # Self-referential FK: parent_id → document_chunks.id
    # ------------------------------------------------------------------
    op.create_table(
        "document_chunks",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("chunk_type", sa.String(10), nullable=False),
        sa.Column(
            "parent_id",
            UUID(as_uuid=True),
            sa.ForeignKey("document_chunks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("page_number", sa.Integer, nullable=True),
        sa.Column("section_heading", sa.String(500), nullable=True),
        sa.Column("source_pdf", sa.String(500), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("idx_chunks_parent_id", "document_chunks", ["parent_id"])

    # ------------------------------------------------------------------
    # chat_messages
    # ------------------------------------------------------------------
    op.create_table(
        "chat_messages",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("idx_messages_session_id", "chat_messages", ["session_id"])

    # ------------------------------------------------------------------
    # retrieval_logs
    # ------------------------------------------------------------------
    op.create_table(
        "retrieval_logs",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "message_id",
            UUID(as_uuid=True),
            sa.ForeignKey("chat_messages.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("enhanced_query", sa.Text, nullable=True),
        sa.Column("hyde_answer", sa.Text, nullable=True),
        sa.Column("retrieved_child_ids", ARRAY(UUID(as_uuid=True)), nullable=True),
        sa.Column("retrieved_parent_ids", ARRAY(UUID(as_uuid=True)), nullable=True),
        sa.Column("reranker_scores", JSONB, nullable=True),
        sa.Column("relevance_verdict", sa.String(20), nullable=True),
        sa.Column("hallucination_verdict", sa.String(20), nullable=True),
        sa.Column("ragas_scores", JSONB, nullable=True),
        sa.Column("total_tokens_used", sa.Integer, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )

    # ------------------------------------------------------------------
    # user_feedback
    # ------------------------------------------------------------------
    op.create_table(
        "user_feedback",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "message_id",
            UUID(as_uuid=True),
            sa.ForeignKey("chat_messages.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("feedback", sa.String(4), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )


def downgrade() -> None:
    op.drop_table("user_feedback")
    op.drop_table("retrieval_logs")
    op.drop_index("idx_messages_session_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("idx_chunks_parent_id", table_name="document_chunks")
    op.drop_table("document_chunks")
