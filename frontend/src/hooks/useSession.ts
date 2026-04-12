import { useEffect } from "react";
import { useChatStore } from "../store/chatStore";

const SESSION_KEY = "medibot_session_id";

function generateSessionId(): string {
  return crypto.randomUUID();
}

/**
 * Initialises (or restores) the active session from localStorage.
 * Returns the current session id.
 *
 * Uses useChatStore.getState() inside the effect so React StrictMode's
 * double-invocation reads fresh store state and never creates duplicates.
 */
export function useSession(): string {
  const activeSessionId = useChatStore((s) => s.activeSessionId);

  useEffect(() => {
    let sid = localStorage.getItem(SESSION_KEY);
    if (!sid) {
      sid = generateSessionId();
      localStorage.setItem(SESSION_KEY, sid);
    }

    // Read fresh state — avoids stale-closure duplicate in StrictMode
    const { sessions, addSession, setActiveSession } = useChatStore.getState();

    if (!sessions.find((s) => s.id === sid)) {
      addSession({
        id: sid,
        title: "New conversation",
        createdAt: new Date(),
        messageCount: 0,
      });
    }

    setActiveSession(sid);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return activeSessionId;
}

/**
 * Creates a brand new session and persists it to localStorage.
 */
export function useNewSession() {
  const { addSession, setActiveSession } = useChatStore();

  return () => {
    const sid = generateSessionId();
    localStorage.setItem(SESSION_KEY, sid);
    addSession({
      id: sid,
      title: "New conversation",
      createdAt: new Date(),
      messageCount: 0,
    });
    setActiveSession(sid);
  };
}
