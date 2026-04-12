import { motion } from "framer-motion";

interface HeaderProps {
  onAdminClick: () => void;
  showAdmin: boolean;
}

export function Header({ onAdminClick, showAdmin }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-subtle bg-navy-900">
      <motion.div
        className="flex items-center gap-3"
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4 }}
      >
        {/* Logo mark */}
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-dark flex items-center justify-center shadow-glow">
          <svg viewBox="0 0 24 24" className="w-5 h-5 text-white fill-current">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z" />
          </svg>
        </div>
        <div>
          <h1 className="text-lg font-bold text-primary leading-none">MediBot v2</h1>
          <p className="text-xs text-secondary leading-none mt-0.5">Medical AI Assistant</p>
        </div>
      </motion.div>

      <div className="flex items-center gap-3">
        <span className="flex items-center gap-1.5 text-xs text-secondary">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          Online
        </span>
        <button
          onClick={onAdminClick}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            showAdmin
              ? "bg-accent text-navy-950 shadow-glow"
              : "bg-navy-700 text-secondary hover:text-primary hover:bg-navy-600"
          }`}
        >
          Admin
        </button>
      </div>
    </header>
  );
}
