import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useChatStore } from "../store/chatStore";

import { useSSEStream } from "../hooks/useSSEStream";
import { MessageBubble } from "./MessageBubble";
import { StatusIndicator } from "./StatusIndicator";
import { InputBar } from "./InputBar";

interface ChatInterfaceProps {
  sessionId: string;
}

export function ChatInterface({ sessionId }: ChatInterfaceProps) {
  const { messagesBySession, statusStep, statusMessage, isStreaming } = useChatStore();
  const { startStream } = useSSEStream();
  const bottomRef = useRef<HTMLDivElement>(null);
  const messages = messagesBySession[sessionId] ?? [];

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (query: string) => {
    const assistantMsgId = crypto.randomUUID();

    // Add user message immediately
    useChatStore.getState().addMessage(sessionId, {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      sources: [],
      isStreaming: false,
      feedback: null,
    });

    void startStream({ sessionId, query, assistantMessageId: assistantMsgId });
  };

  return (
    <div className="flex flex-col h-full">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto flex flex-col gap-4">
          {messages.length === 0 ? (
            <WelcomeScreen onSubmit={handleSubmit} />
          ) : (
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
            </AnimatePresence>
          )}

          {/* Status indicator (shown below last message) */}
          {isStreaming && (
            <div className="pl-1">
              <StatusIndicator step={statusStep} message={statusMessage} />
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      <InputBar onSubmit={handleSubmit} disabled={isStreaming} />
    </div>
  );
}

function WelcomeScreen({ onSubmit }: { onSubmit: (q: string) => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center py-20 text-center gap-4"
    >
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent to-accent-dark flex items-center justify-center shadow-glow-lg mb-2">
        <svg viewBox="0 0 24 24" className="w-8 h-8 text-white fill-current">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z" />
        </svg>
      </div>
      <h2 className="text-xl font-bold text-primary">How can I help you today?</h2>
      <p className="text-sm text-secondary max-w-sm">
        Ask any medical question. I'll search through a comprehensive medical knowledge base and
        provide evidence-based answers with source citations.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-4 w-full max-w-md">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => onSubmit(q)}
            className="text-left p-3 rounded-xl bg-navy-800 border border-subtle text-xs text-secondary hover:text-primary hover:border-accent/40 transition-all cursor-pointer"
          >
            {q}
          </button>
        ))}
      </div>
    </motion.div>
  );
}

const EXAMPLE_QUESTIONS = [
  "What are the symptoms of type 2 diabetes?",
  "How does the immune system fight infections?",
  "What causes hypertension?",
  "Explain the stages of wound healing.",
];
