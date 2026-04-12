import { useCallback } from "react";
import { useChatStore, type SourceCitation } from "../store/chatStore";

// Direct backend URL — bypasses Vite proxy entirely.
// In Docker/production the nginx container handles routing,
// so VITE_API_URL is set to "" at build time.
const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

interface StreamOptions {
  sessionId: string;
  query: string;
  assistantMessageId: string;
}

interface ChatResponse {
  message_id: string;
  session_id: string;
  answer: string;
  sources: SourceCitation[];
}

/**
 * Calls POST /api/chat (JSON endpoint) and simulates word-by-word
 * typing on the frontend so the UX still feels like streaming.
 */
export function useSSEStream() {
  const {
    addMessage,
    appendToken,
    setSources,
    finalizeMessage,
    updateMessageId,
    setStatus,
    setIsStreaming,
  } = useChatStore();

  const startStream = useCallback(
    async ({ sessionId, query, assistantMessageId }: StreamOptions) => {
      setIsStreaming(true);
      setStatus("thinking", "Understanding your question…");

      // Add blank assistant message immediately so the user sees it
      addMessage(sessionId, {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        sources: [],
        isStreaming: true,
        feedback: null,
      });

      const url = `${API_BASE}/api/chat`;
      console.log("[MediBot] POST", url, "| session:", sessionId);

      try {
        setStatus("searching", "Searching medical knowledge base…");

        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, query }),
        });

        console.log("[MediBot] Response status:", response.status);

        if (!response.ok) {
          const errText = await response.text();
          throw new Error(`HTTP ${response.status}: ${errText}`);
        }

        const data: ChatResponse = await response.json();

        // Sync the backend message_id into the store so feedback works correctly.
        // The frontend generated a temp UUID; the DB uses the backend's UUID.
        // Use effectiveId for all subsequent store operations so they find the message.
        const effectiveId = data.message_id ?? assistantMessageId;
        if (data.message_id && data.message_id !== assistantMessageId) {
          updateMessageId(sessionId, assistantMessageId, data.message_id);
        }

        setStatus("generating", "Generating answer…");

        // Simulate word-by-word typing for a smooth UX
        if (data.answer) {
          const words = data.answer.split(" ");
          for (let i = 0; i < words.length; i++) {
            const token = i < words.length - 1 ? words[i] + " " : words[i];
            appendToken(sessionId, effectiveId, token);
            // Small delay between words for typing effect
            await new Promise((resolve) => setTimeout(resolve, 18));
          }
        }

        // Set sources after answer is fully typed
        if (data.sources?.length) {
          setSources(sessionId, effectiveId, data.sources);
        }

        finalizeMessage(sessionId, effectiveId);
        setStatus(null);
        setIsStreaming(false);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[MediBot] Request failed:", msg);
        appendToken(sessionId, assistantMessageId, `⚠️ Error: ${msg}`);
        finalizeMessage(sessionId, assistantMessageId);
        setStatus(null);
        setIsStreaming(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  return { startStream };
}
