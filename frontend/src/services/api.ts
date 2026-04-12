import axios from "axios";

// In Docker/production, VITE_API_URL is set at build time to "".
// In local dev, fall back to the direct backend address.
const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const http = axios.create({ baseURL: BASE });

// ── Feedback ──────────────────────────────────────────────────────────────────

export interface FeedbackPayload {
  message_id: string;
  feedback: "up" | "down";
}

export async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  await http.post("/api/feedback", payload);
}

// ── Admin stats ───────────────────────────────────────────────────────────────

export interface AdminStats {
  total_queries: number;
  avg_latency_ms: number;
  cache_hit_rate: number;
  avg_ragas_scores: Record<string, number>;
  feedback: { up: number; down: number };
  queries_per_day: { date: string; count: number }[];
}

export async function fetchAdminStats(): Promise<AdminStats> {
  const res = await http.get<AdminStats>("/api/admin/stats");
  return res.data;
}

// ── Cache clear ───────────────────────────────────────────────────────────────

export async function clearCache(): Promise<{ cleared_entries: number }> {
  const res = await http.delete<{ cleared_entries: number }>("/api/admin/cache/clear");
  return res.data;
}
