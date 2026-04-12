import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { fetchAdminStats, clearCache, type AdminStats } from "../services/api";

export function AdminPanel() {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [cacheMsg, setCacheMsg] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await fetchAdminStats();
      setStats(data);
    } catch {
      // pass
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const handleClearCache = async () => {
    try {
      const res = await clearCache();
      setCacheMsg(`Cleared ${res.cleared_entries} entries`);
      setTimeout(() => setCacheMsg(null), 3000);
    } catch {
      setCacheMsg("Failed to clear cache");
    }
  };

  const totalFeedback = stats ? stats.feedback.up + stats.feedback.down : 0;
  const upPct = totalFeedback > 0 ? Math.round((stats!.feedback.up / totalFeedback) * 100) : 0;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="w-80 bg-navy-900 border-l border-subtle flex flex-col h-full overflow-y-auto p-4 gap-4"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-primary">Admin Panel</h2>
        <button
          onClick={load}
          className="text-xs text-secondary hover:text-accent transition-colors"
        >
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="flex justify-center py-8">
          <span className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      ) : stats ? (
        <>
          {/* Stat cards */}
          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Total queries" value={stats.total_queries.toString()} />
            <StatCard label="Avg latency" value={`${stats.avg_latency_ms}ms`} />
            <StatCard label="Cache hit rate" value={`${stats.cache_hit_rate}%`} accent="green" />
            <StatCard
              label="Thumbs up"
              value={`${stats.feedback.up} (${upPct}%)`}
              accent="green"
            />
            <StatCard
              label="Thumbs down"
              value={`${stats.feedback.down}`}
              accent="red"
            />
          </div>

          {/* Avg RAGAS scores */}
          {Object.keys(stats.avg_ragas_scores).length > 0 && (
            <div className="bg-navy-800 rounded-xl p-3 border border-subtle">
              <p className="text-xs font-medium text-secondary mb-2">Avg RAGAS scores</p>
              <div className="flex flex-col gap-1.5">
                {Object.entries(stats.avg_ragas_scores).map(([key, val]) => (
                  <div key={key} className="flex items-center justify-between text-xs">
                    <span className="text-secondary capitalize">{key.replace(/_/g, " ")}</span>
                    <span className={`font-medium ${Number(val) >= 0.8 ? "text-green-400" : Number(val) >= 0.6 ? "text-yellow-400" : "text-red-400"}`}>
                      {(Number(val) * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Queries per day */}
          {stats.queries_per_day.length > 0 && (
            <div className="bg-navy-800 rounded-xl p-3 border border-subtle">
              <p className="text-xs font-medium text-secondary mb-2">Queries (last 7 days)</p>
              <div className="flex flex-col gap-1">
                {stats.queries_per_day.map(({ date, count }) => (
                  <div key={date} className="flex items-center justify-between text-xs">
                    <span className="text-secondary">{date}</span>
                    <span className="font-medium text-primary">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Cache management */}
          <div className="bg-navy-800 rounded-xl p-3 border border-subtle">
            <p className="text-xs font-medium text-secondary mb-2">Cache</p>
            <button
              onClick={handleClearCache}
              className="w-full px-3 py-2 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 text-xs font-medium transition-colors border border-red-500/20"
            >
              Clear semantic cache
            </button>
            {cacheMsg && (
              <p className="text-xs text-secondary mt-1.5 text-center">{cacheMsg}</p>
            )}
          </div>
        </>
      ) : (
        <p className="text-xs text-secondary text-center py-4">Failed to load stats.</p>
      )}
    </motion.div>
  );
}

function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "green" | "red";
}) {
  const valueColor =
    accent === "green"
      ? "text-green-400"
      : accent === "red"
        ? "text-red-400"
        : "text-primary";

  return (
    <div className="bg-navy-800 rounded-xl p-3 border border-subtle">
      <p className="text-xs text-secondary mb-1">{label}</p>
      <p className={`text-lg font-bold ${valueColor}`}>{value}</p>
    </div>
  );
}
