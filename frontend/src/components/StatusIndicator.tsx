import { motion, AnimatePresence } from "framer-motion";
import type { StatusStep } from "../store/chatStore";

interface StatusIndicatorProps {
  step: StatusStep;
  message: string;
}

const STEP_ICONS: Record<NonNullable<StatusStep>, string> = {
  thinking: "🧠",
  searching: "🔍",
  analyzing: "📊",
  generating: "✍️",
};

export function StatusIndicator({ step, message }: StatusIndicatorProps) {
  return (
    <AnimatePresence>
      {step && (
        <motion.div
          key={step}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.25 }}
          className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-navy-800 border border-subtle w-fit"
        >
          <span className="text-base">{STEP_ICONS[step]}</span>

          {/* Animated dots */}
          <span className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-dot"
                style={{ animationDelay: `${i * 0.16}s` }}
              />
            ))}
          </span>

          <span className="text-sm text-secondary">{message}</span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
