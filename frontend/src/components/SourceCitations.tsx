import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { SourceCitation } from "../store/chatStore";

interface SourceCitationsProps {
  sources: SourceCitation[];
}

export function SourceCitations({ sources }: SourceCitationsProps) {
  const [expanded, setExpanded] = useState(false);

  if (!sources.length) return null;

  return (
    <div className="mt-3">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-2 text-xs text-accent hover:text-accent-light transition-colors"
      >
        <svg
          className={`w-3 h-3 transition-transform ${expanded ? "rotate-90" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2.5}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
        {sources.length} source{sources.length !== 1 ? "s" : ""}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="mt-2 flex flex-col gap-2">
              {sources.map((src, idx) => (
                <div
                  key={src.chunk_id}
                  className="p-3 rounded-lg bg-navy-700 border border-subtle text-xs"
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="w-5 h-5 rounded-full bg-navy-600 flex items-center justify-center text-accent font-bold text-[10px]">
                      {idx + 1}
                    </span>
                    <span className="text-accent font-medium truncate">
                      {src.section_heading || "Section"}
                    </span>
                    <span className="ml-auto text-secondary shrink-0">p.{src.page_number}</span>
                  </div>
                  <p className="text-secondary leading-relaxed line-clamp-3">{src.excerpt}</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
