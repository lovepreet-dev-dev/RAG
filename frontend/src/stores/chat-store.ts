"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ChatMessage, SourceDocument } from "@/types";

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  isStreaming: boolean;
  streamingController: AbortController | null;

  createSession: () => string;
  setActiveSession: (id: string | null) => void;
  deleteSession: (id: string) => void;
  renameSession: (id: string, title: string) => void;
  addMessage: (sessionId: string, message: ChatMessage) => void;
  updateLastAssistant: (sessionId: string, content: string) => void;
  appendToLastAssistant: (sessionId: string, token: string) => void;
  setSources: (sessionId: string, sources: SourceDocument[]) => void;
  setStreaming: (v: boolean) => void;
  setStreamingController: (c: AbortController | null) => void;
  getActiveSession: () => ChatSession | undefined;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      isStreaming: false,
      streamingController: null,

      createSession: () => {
        const id = crypto.randomUUID();
        const now = new Date().toISOString();
        const session: ChatSession = {
          id,
          title: "New Chat",
          messages: [],
          created_at: now,
          updated_at: now,
        };
        set((s) => ({
          sessions: [session, ...s.sessions],
          activeSessionId: id,
        }));
        return id;
      },

      setActiveSession: (id) => set({ activeSessionId: id }),

      deleteSession: (id) =>
        set((s) => {
          const filtered = s.sessions.filter((x) => x.id !== id);
          return {
            sessions: filtered,
            activeSessionId:
              s.activeSessionId === id
                ? filtered[0]?.id ?? null
                : s.activeSessionId,
          };
        }),

      renameSession: (id, title) =>
        set((s) => ({
          sessions: s.sessions.map((x) =>
            x.id === id ? { ...x, title, updated_at: new Date().toISOString() } : x,
          ),
        })),

      addMessage: (sessionId, message) =>
        set((s) => ({
          sessions: s.sessions.map((x) =>
            x.id === sessionId
              ? {
                  ...x,
                  messages: [...x.messages, message],
                  updated_at: new Date().toISOString(),
                }
              : x,
          ),
        })),

      updateLastAssistant: (sessionId, content) =>
        set((s) => ({
          sessions: s.sessions.map((x) => {
            if (x.id !== sessionId) return x;
            const msgs = [...x.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role === "assistant") {
              msgs[msgs.length - 1] = { ...last, content };
            }
            return { ...x, messages: msgs, updated_at: new Date().toISOString() };
          }),
        })),

      appendToLastAssistant: (sessionId, token) =>
        set((s) => ({
          sessions: s.sessions.map((x) => {
            if (x.id !== sessionId) return x;
            const msgs = [...x.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role === "assistant") {
              msgs[msgs.length - 1] = { ...last, content: last.content + token };
            }
            return { ...x, messages: msgs };
          }),
        })),

      setSources: (sessionId, sources) =>
        set((s) => ({
          sessions: s.sessions.map((x) => {
            if (x.id !== sessionId) return x;
            const msgs = [...x.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role === "assistant") {
              msgs[msgs.length - 1] = { ...last, sources };
            }
            return { ...x, messages: msgs };
          }),
        })),

      setStreaming: (v) => set({ isStreaming: v }),
      setStreamingController: (c) => set({ streamingController: c }),

      getActiveSession: () => {
        const s = get();
        return s.sessions.find((x) => x.id === s.activeSessionId);
      },
    }),
    {
      name: "rag-chat",
      partialize: (s) => ({
        sessions: s.sessions,
        activeSessionId: s.activeSessionId,
      }),
    },
  ),
);
