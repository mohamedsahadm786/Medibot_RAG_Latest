import { motion } from "framer-motion";
import { useChatStore, type Session } from "../store/chatStore";
import { useNewSession } from "../hooks/useSession";

export function ChatSidebar() {
  const { sessions, activeSessionId, setActiveSession, messagesBySession } = useChatStore();
  const createNewSession = useNewSession();

  const getSessionTitle = (session: Session): string => {
    const msgs = messagesBySession[session.id] ?? [];
    const firstUser = msgs.find((m) => m.role === "user");
    if (firstUser) {
      return firstUser.content.length > 36
        ? firstUser.content.slice(0, 36) + "…"
        : firstUser.content;
    }
    return session.title;
  };

  const formatDate = (d: Date): string => {
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 60_000) return "Just now";
    if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
    return d.toLocaleDateString();
  };

  return (
    <aside className="w-64 bg-navy-900 border-r border-subtle flex flex-col h-full">
      <div className="p-4 border-b border-subtle">
        <button
          onClick={createNewSession}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl bg-navy-700 hover:bg-navy-600 text-sm text-primary transition-colors"
        >
          <svg className="w-4 h-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          New conversation
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {sessions.length === 0 ? (
          <p className="text-xs text-secondary text-center mt-8 px-4">No conversations yet</p>
        ) : (
          <div className="flex flex-col gap-1">
            {sessions.map((session) => (
              <motion.button
                key={session.id}
                whileTap={{ scale: 0.98 }}
                onClick={() => setActiveSession(session.id)}
                className={`w-full text-left px-3 py-2.5 rounded-xl text-sm transition-all ${
                  activeSessionId === session.id
                    ? "bg-accent/15 text-primary border border-accent/30"
                    : "text-secondary hover:bg-navy-700 hover:text-primary"
                }`}
              >
                <p className="font-medium truncate">{getSessionTitle(session)}</p>
                <p className="text-xs opacity-50 mt-0.5">{formatDate(session.createdAt)}</p>
              </motion.button>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
