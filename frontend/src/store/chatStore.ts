import { create } from "zustand";

export interface SourceCitation {
  chunk_id: string;
  page_number: number;
  section_heading: string;
  excerpt: string;
  source_pdf: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: SourceCitation[];
  isStreaming: boolean;
  feedback: "up" | "down" | null;
}

export interface Session {
  id: string;
  title: string;
  createdAt: Date;
  messageCount: number;
}

export type StatusStep = "thinking" | "searching" | "analyzing" | "generating" | null;

interface ChatStore {
  // Sessions
  sessions: Session[];
  activeSessionId: string;
  addSession: (session: Session) => void;
  setActiveSession: (id: string) => void;

  // Messages (keyed by session id)
  messagesBySession: Record<string, Message[]>;
  addMessage: (sessionId: string, message: Message) => void;
  appendToken: (sessionId: string, messageId: string, token: string) => void;
  setSources: (sessionId: string, messageId: string, sources: SourceCitation[]) => void;
  finalizeMessage: (sessionId: string, messageId: string) => void;
  updateMessageId: (sessionId: string, oldId: string, newId: string) => void;
  setMessageFeedback: (sessionId: string, messageId: string, feedback: "up" | "down") => void;

  // Streaming status
  statusStep: StatusStep;
  statusMessage: string;
  setStatus: (step: StatusStep, message?: string) => void;
  isStreaming: boolean;
  setIsStreaming: (value: boolean) => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  sessions: [],
  activeSessionId: "",
  addSession: (session) =>
    set((state) => ({ sessions: [session, ...state.sessions] })),
  setActiveSession: (id) => set({ activeSessionId: id }),

  messagesBySession: {},
  addMessage: (sessionId, message) =>
    set((state) => ({
      messagesBySession: {
        ...state.messagesBySession,
        [sessionId]: [...(state.messagesBySession[sessionId] ?? []), message],
      },
    })),
  appendToken: (sessionId, messageId, token) =>
    set((state) => {
      const msgs = state.messagesBySession[sessionId] ?? [];
      return {
        messagesBySession: {
          ...state.messagesBySession,
          [sessionId]: msgs.map((m) =>
            m.id === messageId ? { ...m, content: m.content + token } : m,
          ),
        },
      };
    }),
  setSources: (sessionId, messageId, sources) =>
    set((state) => {
      const msgs = state.messagesBySession[sessionId] ?? [];
      return {
        messagesBySession: {
          ...state.messagesBySession,
          [sessionId]: msgs.map((m) =>
            m.id === messageId ? { ...m, sources } : m,
          ),
        },
      };
    }),
  finalizeMessage: (sessionId, messageId) =>
    set((state) => {
      const msgs = state.messagesBySession[sessionId] ?? [];
      return {
        messagesBySession: {
          ...state.messagesBySession,
          [sessionId]: msgs.map((m) =>
            m.id === messageId ? { ...m, isStreaming: false } : m,
          ),
        },
      };
    }),
  updateMessageId: (sessionId, oldId, newId) =>
    set((state) => {
      const msgs = state.messagesBySession[sessionId] ?? [];
      return {
        messagesBySession: {
          ...state.messagesBySession,
          [sessionId]: msgs.map((m) =>
            m.id === oldId ? { ...m, id: newId } : m,
          ),
        },
      };
    }),
  setMessageFeedback: (sessionId, messageId, feedback) =>
    set((state) => {
      const msgs = state.messagesBySession[sessionId] ?? [];
      return {
        messagesBySession: {
          ...state.messagesBySession,
          [sessionId]: msgs.map((m) =>
            m.id === messageId ? { ...m, feedback } : m,
          ),
        },
      };
    }),

  statusStep: null,
  statusMessage: "",
  setStatus: (step, message = "") => set({ statusStep: step, statusMessage: message }),
  isStreaming: false,
  setIsStreaming: (value) => set({ isStreaming: value }),
}));
