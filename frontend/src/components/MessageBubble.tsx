import { motion } from "framer-motion";
import type { Message } from "../store/chatStore";
import { SourceCitations } from "./SourceCitations";
import { FeedbackButtons } from "./FeedbackButtons";

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex ${isUser ? "justify-end" : "justify-start"} animate-fade-in`}
    >
      <div className={`max-w-[75%] ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        {/* Bubble */}
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? "bg-accent text-navy-950 rounded-tr-sm"
              : "bg-navy-800 text-primary border border-subtle rounded-tl-sm"
          }`}
        >
          {message.content}
          {/* Streaming cursor */}
          {message.isStreaming && (
            <span className="inline-block w-0.5 h-4 bg-accent ml-0.5 animate-cursor align-middle" />
          )}
        </div>

        {/* Sources (assistant only) */}
        {!isUser && !message.isStreaming && message.sources.length > 0 && (
          <div className="w-full px-1">
            <SourceCitations sources={message.sources} />
          </div>
        )}

        {/* Feedback (assistant only, after streaming) */}
        {!isUser && !message.isStreaming && (
          <div className="px-1">
            <FeedbackButtons messageId={message.id} currentFeedback={message.feedback} />
          </div>
        )}
      </div>
    </motion.div>
  );
}
