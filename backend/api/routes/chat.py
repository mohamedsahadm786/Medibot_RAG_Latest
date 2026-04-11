from fastapi import APIRouter

router = APIRouter()


# Placeholder — full implementation in Phase 3 (basic retrieval + generation)
# Phase 8 converts this to SSE streaming
@router.post("/chat")
async def chat_placeholder() -> dict:
    """Chat endpoint — implemented in Phase 3."""
    return {"detail": "Chat endpoint not yet implemented. Coming in Phase 3."}
