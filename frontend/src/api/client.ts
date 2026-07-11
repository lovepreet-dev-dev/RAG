import axios, { AxiosError, InternalAxiosRequestConfig } from "axios";
import type {
  AuthResponse,
  ChatRequest,
  ChatRequest as ChatReq,
  CreateUserRequest,
  DocumentListResponse,
  FeedbackRequest,
  HealthResponse,
  LoginRequest,
  QueryResponse,
  Settings,
  StatsResponse,
  StreamChunk,
  UploadResponse,
  User,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Axios instance ──────────────────────────────────────────────────────────

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 120_000,
  headers: { "Content-Type": "application/json" },
});

// ── Token refresh state ─────────────────────────────────────────────────────

let isRefreshing = false;
let failedQueue: Array<{
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
}> = [];

const processQueue = (error: unknown, token: string | null = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

// ── Auth interceptor ────────────────────────────────────────────────────────

api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("access_token");
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (err: AxiosError<{ detail?: string }>) => {
    const originalRequest = err.config as InternalAxiosRequestConfig & {
      _retry?: boolean;
    };

    // If 401 and not already retrying, attempt token refresh
    if (err.response?.status === 401 && !originalRequest._retry) {
      const refreshToken = localStorage.getItem("refresh_token");

      if (refreshToken) {
        if (isRefreshing) {
          // Queue this request while refresh is in progress
          return new Promise((resolve, reject) => {
            failedQueue.push({ resolve, reject });
          })
            .then((token) => {
              if (originalRequest.headers) {
                originalRequest.headers.Authorization = `Bearer ${token}`;
              }
              return api(originalRequest);
            })
            .catch((err) => Promise.reject(err));
        }

        originalRequest._retry = true;
        isRefreshing = true;

        try {
          const response = await axios.post(`${API_BASE}/api/v1/auth/refresh`, {
            refresh_token: refreshToken,
          });

          const { access_token, refresh_token: newRefreshToken } =
            response.data;

          localStorage.setItem("access_token", access_token);
          localStorage.setItem("refresh_token", newRefreshToken);

          // Update cookies to stay in sync with new tokens
          const secure = typeof window !== "undefined" && !window.location.hostname.includes("localhost");
          const secureFlag = secure ? "; Secure" : "";
          document.cookie = `auth_token=${encodeURIComponent(access_token)}; path=/; max-age=${30 * 60}; SameSite=Lax${secureFlag}`;
          document.cookie = `refresh_token=${encodeURIComponent(newRefreshToken)}; path=/; max-age=${7 * 24 * 60 * 60}; SameSite=Lax${secureFlag}`;

          processQueue(null, access_token);

          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${access_token}`;
          }
          return api(originalRequest);
        } catch (refreshError) {
          processQueue(refreshError, null);
          // Refresh failed, clear tokens and redirect to login
          localStorage.removeItem("access_token");
          localStorage.removeItem("refresh_token");
          localStorage.removeItem("user");
          if (typeof window !== "undefined") {
            window.location.href = "/login";
          }
          return Promise.reject(refreshError);
        } finally {
          isRefreshing = false;
        }
      }

      // No refresh token, redirect to login
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
      localStorage.removeItem("user");
      if (typeof window !== "undefined") {
        window.location.href = "/login";
      }
    }
    return Promise.reject(err);
  },
);

// ── Auth API ────────────────────────────────────────────────────────────────

export const authApi = {
  login: (data: LoginRequest) =>
    api.post<AuthResponse>("/api/v1/auth/login", data).then((r) => r.data),

  me: () => api.get<User>("/api/v1/auth/me").then((r) => r.data),

  logout: (refreshToken: string) =>
    api
      .post("/api/v1/auth/logout", { refresh_token: refreshToken })
      .then((r) => r.data),

  changePassword: (data: { current_password: string; new_password: string }) =>
    api.post("/api/v1/auth/change-password", data).then((r) => r.data),

  register: (data: {
    email: string;
    name: string;
    password: string;
    role?: string;
  }) => api.post<User>("/api/v1/auth/register", data).then((r) => r.data),

  forgotPassword: (email: string) =>
    api.post("/api/v1/auth/forgot-password", { email }).then((r) => r.data),

  resetPassword: (token: string, newPassword: string) =>
    api.post("/api/v1/auth/reset-password", { token, new_password: newPassword }).then((r) => r.data),
};

// ── Chat API ────────────────────────────────────────────────────────────────

export const chatApi = {
  query: (data: ChatRequest) =>
    api.post<QueryResponse>("/api/v1/query", data).then((r) => r.data),

  chat: (data: ChatReq) =>
    api.post<QueryResponse>("/api/v1/chat", { ...data, stream: false }).then((r) => r.data),
};

// ── Streaming ───────────────────────────────────────────────────────────────

export function streamChat(
  data: ChatReq,
  onToken: (chunk: StreamChunk) => void,
  onDone: () => void,
  onError: (error: Error) => void,
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const token = typeof window !== "undefined"
        ? localStorage.getItem("access_token")
        : null;

      const res = await fetch(`${API_BASE}/api/v1/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ ...data, stream: true }),
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const payload = line.slice(6);
            if (payload === "[DONE]") {
              onDone();
              return;
            }
            try {
              const chunk: StreamChunk = JSON.parse(payload);
              onToken(chunk);
            } catch {
              // skip malformed chunks
            }
          }
        }
      }
      onDone();
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onError(err as Error);
      }
    }
  })();

  return controller;
}

// ── Chat History API ─────────────────────────────────────────────────────

export interface ChatSessionResponse {
  id: string;
  title: string;
  is_pinned: boolean;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatMessageResponse {
  id: string;
  role: string;
  content: string;
  query_id: string | null;
  sources: Record<string, unknown>[] | null;
  confidence_score: number | null;
  confidence_level: string | null;
  model_used: string | null;
  tokens_used: number | null;
  latency_ms: number | null;
  created_at: string;
}

export interface SessionDetailResponse extends ChatSessionResponse {
  messages: ChatMessageResponse[];
}

export const chatHistoryApi = {
  listSessions: (limit = 50, offset = 0) =>
    api
      .get<{ sessions: ChatSessionResponse[]; total_count: number }>(
        `/api/v1/chat/sessions?limit=${limit}&offset=${offset}`
      )
      .then((r) => r.data),

  createSession: (title = "New Chat") =>
    api
      .post<ChatSessionResponse>("/api/v1/chat/sessions", { title })
      .then((r) => r.data),

  getSession: (sessionId: string) =>
    api
      .get<SessionDetailResponse>(`/api/v1/chat/sessions/${sessionId}`)
      .then((r) => r.data),

  updateSession: (sessionId: string, data: { title?: string; is_pinned?: boolean }) =>
    api
      .put<ChatSessionResponse>(`/api/v1/chat/sessions/${sessionId}`, data)
      .then((r) => r.data),

  deleteSession: (sessionId: string) =>
    api
      .delete(`/api/v1/chat/sessions/${sessionId}`)
      .then((r) => r.data),

  addMessage: (
    sessionId: string,
    data: {
      role: string;
      content: string;
      query_id?: string;
      sources?: Record<string, unknown>[];
      confidence_score?: number;
      confidence_level?: string;
      model_used?: string;
      tokens_used?: number;
      latency_ms?: number;
    }
  ) =>
    api
      .post<ChatMessageResponse>(`/api/v1/chat/sessions/${sessionId}/messages`, data)
      .then((r) => r.data),
};

// ── Documents API ───────────────────────────────────────────────────────────

export const documentsApi = {
  list: () =>
    api.get<DocumentListResponse>("/api/v1/documents").then((r) => r.data),

  upload: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return api
      .post<UploadResponse>("/api/v1/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },

  ingest: (filenames: string[]) =>
    api
      .post<UploadResponse>("/api/v1/ingest", { filenames })
      .then((r) => r.data),

  delete: (documentId: string) =>
    api.delete(`/api/v1/documents/${documentId}`).then((r) => r.data),

  getContent: (documentId: string) =>
    api.get(`/api/v1/documents/${documentId}/content`).then((r) => r.data),

  download: (documentId: string) => {
    const token = typeof window !== "undefined"
      ? localStorage.getItem("access_token")
      : null;
    return fetch(`${API_BASE}/api/v1/documents/${documentId}/download`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    }).then((r) => r.blob());
  },

  getStats: () =>
    api.get<StatsResponse>("/api/v1/stats").then((r) => r.data),

  health: () =>
    api.get<HealthResponse>("/api/v1/health").then((r) => r.data),
};

// ── Settings API ────────────────────────────────────────────────────────────

export const settingsApi = {
  get: () => api.get<Settings>("/api/v1/settings").then((r) => r.data),

  update: (data: Partial<Settings>) =>
    api.put<Settings>("/api/v1/settings", data).then((r) => r.data),
};

// ── Users API ───────────────────────────────────────────────────────────────

export const usersApi = {
  list: () =>
    api.get<{ users: User[]; total_count: number }>("/api/v1/auth/users").then((r) => r.data.users),

  create: (data: CreateUserRequest) =>
    api.post<User>("/api/v1/auth/users", data).then((r) => r.data),

  update: (id: string, data: Partial<User>) =>
    api.put<User>(`/api/v1/auth/users/${id}`, data).then((r) => r.data),

  delete: (id: string) =>
    api.delete(`/api/v1/auth/users/${id}`).then((r) => r.data),
};

// ── Feedback API ────────────────────────────────────────────────────────────

export const feedbackApi = {
  submit: (data: FeedbackRequest) =>
    api.post("/api/v1/feedback", data).then((r) => r.data),
};
