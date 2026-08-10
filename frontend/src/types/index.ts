// ── Auth Types ──────────────────────────────────────────────────────────────

export type Role = "admin" | "employee" | "viewer";

export interface User {
  id: string;
  email: string;
  name: string;
  role: Role;
  avatar?: string;
  created_at: string;
  last_login?: string;
  is_active: boolean;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface CreateUserRequest {
  email: string;
  name: string;
  password: string;
  role: Role;
}

// ── Document Types ──────────────────────────────────────────────────────────

export interface IngestedDocument {
  document_id: string;
  filename: string;
  total_chunks: number;
  total_pages: number;
  created_at: string | null;
}

export interface DocumentDetail extends IngestedDocument {
  embedding_model?: string;
  status: "indexed" | "processing" | "error";
  file_size?: number;
  content_type?: string;
}

export interface UploadResponse {
  documents: IngestedDocument[];
  total_chunks: number;
  message: string;
}

export interface DocumentListResponse {
  documents: IngestedDocument[];
  total_count: number;
}

// ── Query Types ─────────────────────────────────────────────────────────────

export interface SourceDocument {
  content: string;
  filename: string;
  page_number: number;
  chunk_id: number;
  document_id: string;
  score: number | null;
}

export interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
  query_id: string;
  confidence_score: number;
  confidence_level: "low" | "medium" | "high";
  model_used: string;
  tokens_used: number;
  cost_estimate: number;
  latency_ms: number;
  session_id?: string;
  query_type: string;
  validation?: {
    is_valid: boolean;
    citation_score: number;
    coverage_score: number;
    hallucination_score: number;
    overall_score: number;
    issues: string[];
    unsupported_statements: string[];
  };
}

export interface StreamChunk {
  token: string;
  query_id: string;
  is_final: boolean;
  answer?: string;
  sources?: SourceDocument[];
  confidence_score?: number;
  confidence_level?: string;
  model_used?: string;
  metadata?: Record<string, unknown>;
}

// ── Chat Types ──────────────────────────────────────────────────────────────

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceDocument[];
  query_id?: string;
  confidence_score?: number;
  confidence_level?: "low" | "medium" | "high";
  model_used?: string;
  tokens_used?: number;
  latency_ms?: number;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  top_k?: number;
  retrieval_mode?: string;
  enable_reranking?: boolean;
  stream?: boolean;
}

// ── Stats Types ─────────────────────────────────────────────────────────────

export interface StatsResponse {
  total_documents: number;
  total_chunks: number;
  total_queries: number;
  total_tokens_used: number;
  total_cost: number;
  vector_store_type: string;
  embedding_model: string;
  llm_provider: string;
  avg_latency_ms: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  vector_store: string;
  embedding_model: string;
  llm_provider: string;
  documents_indexed: number;
}

// ── Settings Types ──────────────────────────────────────────────────────────

export interface Settings {
  llm_provider: string;
  openai_model: string;
  anthropic_model: string;
  openrouter_model: string;
  embedding_model: string;
  vector_store: string;
  enable_hybrid_search: boolean;
  enable_reranking: boolean;
  enable_answer_validation: boolean;
  similarity_top_k: number;
  hybrid_top_k: number;
  chunk_size: number;
  chunk_overlap: number;
  max_context_tokens: number;
  score_threshold: number;
  rerank_top_k: number;
  temperature: number;
}

export interface AppSettings extends Settings {}

// ── Feedback Types ──────────────────────────────────────────────────────────

export interface FeedbackRequest {
  query_id: string;
  rating: number;
  comment?: string;
}

// ── API Error ───────────────────────────────────────────────────────────────

export interface ApiError {
  detail: string;
  error_code?: string;
}
