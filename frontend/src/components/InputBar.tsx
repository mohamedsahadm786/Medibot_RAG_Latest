import { useState, useRef, type KeyboardEvent } from "react";
import { motion } from "framer-motion";

interface InputBarProps {
  onSubmit: (query: string) => void;
  disabled: boolean;
}

export function InputBar({ onSubmit, disabled }: InputBarProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="px-4 py-4 border-t border-subtle bg-navy-900">
      <div className="max-w-3xl mx-auto">
        <div
          className={`flex items-end gap-3 p-3 rounded-2xl border transition-all ${
            disabled
              ? "border-subtle bg-navy-800"
              : "border-navy-700 bg-navy-800 focus-within:border-accent focus-within:shadow-glow"
          }`}
        >
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            disabled={disabled}
            placeholder="Ask a medical question…"
            rows={1}
            className="flex-1 bg-transparent text-sm text-primary placeholder-secondary resize-none outline-none min-h-[24px] max-h-40 leading-relaxed"
          />
          <motion.button
            whileTap={{ scale: 0.92 }}
            onClick={handleSubmit}
            disabled={disabled || !value.trim()}
            className={`shrink-0 w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
              disabled || !value.trim()
                ? "bg-navy-700 text-secondary cursor-not-allowed"
                : "bg-accent text-navy-950 shadow-glow hover:bg-accent-light"
            }`}
          >
            {disabled ? (
              <span className="w-3 h-3 border-2 border-secondary border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
              </svg>
            )}
          </motion.button>
        </div>
        <p className="text-center text-xs text-secondary mt-2 opacity-50">
          For informational purposes only. Always consult a medical professional.
        </p>
      </div>
    </div>
  );
}
