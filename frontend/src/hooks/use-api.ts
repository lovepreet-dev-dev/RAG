"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { authApi, chatApi, chatHistoryApi, documentsApi, feedbackApi, settingsApi, usersApi } from "@/api/client";
import type { ChatRequest, CreateUserRequest, FeedbackRequest, Settings } from "@/types";
import { useAuthStore } from "@/stores/auth-store";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

// ── Auth ────────────────────────────────────────────────────────────────────

export function useLogin() {
  const { login } = useAuthStore();
  const router = useRouter();

  return useMutation({
    mutationFn: authApi.login,
    onSuccess: (data) => {
      // Fetch user info after getting tokens
      authApi.me().then((user) => {
        login(user, data.access_token, data.refresh_token);
        toast.success("Welcome back!");
        router.push("/chat");
      }).catch(() => {
        // If me() fails (e.g., auth disabled), use a default user
        login(
          {
            id: "default",
            email: "admin@example.com",
            name: "Admin",
            role: "admin",
            is_active: true,
            created_at: new Date().toISOString(),
          },
          data.access_token,
          data.refresh_token,
        );
        toast.success("Welcome back!");
        router.push("/chat");
      });
    },
    onError: (err: Error) => {
      toast.error(err.message || "Login failed");
    },
  });
}

// ── Chat ────────────────────────────────────────────────────────────────────

export function useChatQuery() {
  return useMutation({
    mutationFn: chatApi.query,
  });
}

// ── Documents ───────────────────────────────────────────────────────────────

export function useDocuments() {
  return useQuery({
    queryKey: ["documents"],
    queryFn: documentsApi.list,
    staleTime: 30_000,
  });
}

export function useUploadDocuments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (files: File[]) => documentsApi.upload(files),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      toast.success("Documents uploaded successfully");
    },
    onError: (err: Error) => {
      toast.error(err.message || "Upload failed");
    },
  });
}

export function useIngestDocuments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (filenames: string[]) => documentsApi.ingest(filenames),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      toast.success("Documents indexed successfully");
    },
    onError: (err: Error) => {
      toast.error(err.message || "Ingestion failed");
    },
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => documentsApi.delete(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      toast.success("Document deleted");
    },
    onError: (err: Error) => {
      toast.error(err.message || "Delete failed");
    },
  });
}

// ── Stats ───────────────────────────────────────────────────────────────────

export function useStats() {
  return useQuery({
    queryKey: ["stats"],
    queryFn: documentsApi.getStats,
    staleTime: 60_000,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: documentsApi.health,
    refetchInterval: 30_000,
  });
}

// ── Settings ────────────────────────────────────────────────────────────────

export function useSettings() {
  return useQuery({
    queryKey: ["settings"],
    queryFn: settingsApi.get,
  });
}

export function useUpdateSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<Settings>) => settingsApi.update(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["settings"] });
      toast.success("Settings saved");
    },
    onError: (err: Error) => {
      toast.error(err.message || "Failed to save settings");
    },
  });
}

// ── Users ───────────────────────────────────────────────────────────────────

export function useUsers() {
  return useQuery({
    queryKey: ["users"],
    queryFn: usersApi.list,
  });
}

export function useCreateUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateUserRequest) => usersApi.create(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["users"] });
      toast.success("User created");
    },
    onError: (err: Error) => {
      toast.error(err.message || "Failed to create user");
    },
  });
}

export function useDeleteUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => usersApi.delete(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["users"] });
      toast.success("User deleted");
    },
    onError: (err: Error) => {
      toast.error(err.message || "Failed to delete user");
    },
  });
}

// ── Feedback ────────────────────────────────────────────────────────────────

export function useSubmitFeedback() {
  return useMutation({
    mutationFn: (data: FeedbackRequest) => feedbackApi.submit(data),
    onError: () => {
      toast.error("Failed to submit feedback");
    },
  });
}

// ── Chat History ──────────────────────────────────────────────────────────

export function useChatSessions() {
  return useQuery({
    queryKey: ["chat-sessions"],
    queryFn: () => chatHistoryApi.listSessions(),
    staleTime: 10_000,
  });
}

export function useCreateChatSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (title?: string) => chatHistoryApi.createSession(title),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["chat-sessions"] });
    },
  });
}

export function useChatSession(sessionId: string | null) {
  return useQuery({
    queryKey: ["chat-session", sessionId],
    queryFn: () => chatHistoryApi.getSession(sessionId!),
    enabled: !!sessionId,
  });
}

export function useUpdateChatSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ sessionId, ...data }: { sessionId: string; title?: string; is_pinned?: boolean }) =>
      chatHistoryApi.updateSession(sessionId, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["chat-sessions"] });
    },
  });
}

export function useDeleteChatSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => chatHistoryApi.deleteSession(sessionId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["chat-sessions"] });
    },
  });
}

export function useAddChatMessage() {
  return useMutation({
    mutationFn: ({
      sessionId,
      ...data
    }: {
      sessionId: string;
      role: string;
      content: string;
      query_id?: string;
      sources?: Record<string, unknown>[];
      confidence_score?: number;
      confidence_level?: string;
      model_used?: string;
      tokens_used?: number;
      latency_ms?: number;
    }) => chatHistoryApi.addMessage(sessionId, data),
  });
}
