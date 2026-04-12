"""
MediBot v2 — Generator Service (Phase 3)

Builds a medical-domain prompt from retrieved context chunks
and calls GPT-4o to generate a grounded answer.
"""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.config import settings

logger = logging.getLogger(__name__)

# ── Medical system prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a medical knowledge assistant with access to a \
comprehensive medical reference encyclopedia.

Guidelines:
- Answer medical questions accurately and only based on the provided context.
- If the context does not contain enough information to answer a medical question, say: \
"I could not find sufficient information about this topic in the medical reference."
- Always cite the section and page number your answer comes from when using medical context.
- For conversational or personal questions (e.g. greetings, user sharing their name), \
use the conversation history to respond naturally — do not search the medical reference.
- Be clear, structured, and use plain language where possible.
- This information is for educational purposes only — not a substitute for \
professional medical advice, diagnosis, or treatment."""

STRICT_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nCRITICAL: Every single claim in your answer MUST be explicitly stated "
    "in the provided context. Do not infer, extrapolate, or add any information "
    "that is not directly present in the context passages."
)

# ── Prompt templates ───────────────────────────────────────────────────────────
_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Conversation History:\n{summaries}\n\nMedical Reference Context:\n{context}\n\nQuestion: {query}",
        ),
    ]
)

_STRICT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", STRICT_SYSTEM_PROMPT),
        (
            "human",
            "Conversation History:\n{summaries}\n\nMedical Reference Context:\n{context}\n\nQuestion: {query}",
        ),
    ]
)

# ── LLM singleton ──────────────────────────────────────────────────────────────
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """Instantiate ChatOpenAI once and reuse across requests."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.openai_model_primary,
            temperature=0.1,
            api_key=settings.openai_api_key,
        )
    return _llm


# ── Context builder ────────────────────────────────────────────────────────────

def _build_context(chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved parent chunks into a numbered context block.

    Each chunk is labelled with its source (page + section) so the
    LLM can cite it naturally in the answer.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[Source {i}] Page {chunk['page_number']}"
            + (f" — {chunk['section_heading']}" if chunk["section_heading"] else "")
        )
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


# ── Main generation function ───────────────────────────────────────────────────

async def generate(
    query: str,
    chunks: list[dict[str, Any]],
    strict: bool = False,
    summaries: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate a grounded medical answer from retrieved context.

    Args:
        query:     The user's question.
        chunks:    Parent chunks returned by the retriever.
        strict:    Use the stricter hallucination-resistant prompt.
        summaries: Conversation turn summaries for memory context.

    Returns:
        {
            "answer": str,
            "sources": [{"chunk_id", "page_number", "section_heading",
                         "excerpt", "source_pdf"}, ...]
        }
    """
    summaries_text = (
        "\n".join(f"- {s}" for s in summaries) if summaries else "None"
    )

    if not chunks:
        # No RAG context — let the LLM answer from conversation history alone
        llm = _get_llm()
        no_context_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Conversation History:\n{summaries}\n\nMedical Reference Context:\nNone\n\nQuestion: {query}"),
        ])
        chain = no_context_prompt | llm
        response = await chain.ainvoke({"summaries": summaries_text, "query": query})
        return {"answer": response.content, "sources": []}

    context = _build_context(chunks)
    llm = _get_llm()
    chain = (_STRICT_PROMPT if strict else _PROMPT) | llm

    logger.info(
        "Calling %s with %d context chunks for query: %s",
        settings.openai_model_primary,
        len(chunks),
        query[:80],
    )

    response = await chain.ainvoke({"context": context, "query": query, "summaries": summaries_text})
    answer: str = response.content

    # Build source citations from the chunks used
    sources = [
        {
            "chunk_id": c["chunk_id"],
            "page_number": c["page_number"],
            "section_heading": c["section_heading"],
            "excerpt": c["content"][:300] + ("..." if len(c["content"]) > 300 else ""),
            "source_pdf": c["source_pdf"],
        }
        for c in chunks
    ]

    logger.info("Generation complete (%d chars)", len(answer))
    return {"answer": answer, "sources": sources}
