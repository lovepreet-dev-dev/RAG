"""Pydantic models for API requests, responses, and internal data structures.

Every API boundary uses these schemas for serialisation, validation, and
documentation.  Internal services may use plain dataclasses or dicts, but
the API layer always speaks schemas.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class ChunkingStrategyEnum(str, Enum):
    """Chunking strategies exposed via the API."""

    RECURSIVE = "recursive"


class RetrievalModeEnum(str, Enum):
    """Retrieval modes exposed via the API."""

    SIMILARITY = "similarity"
    HYBRID = "hybrid"


# ── Document Models ──────────────────────────────────────────────────────────


class DocumentMetadata(BaseModel):
    """Metadata attached to every document chunk."""

    filename: str = ""
    page_number: int = 0
    chunk_id: str = ""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    content_type: str = ""
    total_chunks: int = 0


class DocumentChunk(BaseModel):
    """A single chunk of text with metadata ready for embedding."""

    content: str
    metadata: DocumentMetadata
    embedding: Optional[list[float]] = None


class IngestedDocument(BaseModel):
    """Summary of an ingested document."""

    document_id: str
    filename: str
    total_chunks: int
    total_pages: int
    created_at: Union[datetime, str, None] = None


# ── API Request Models ───────────────────────────────────────────────────────


class UploadRequest(BaseModel):
    """Metadata sent alongside file uploads."""

    filename: Optional[str] = None
    content_type: Optional[str] = None


class IngestRequest(BaseModel):
    """Request to trigger ingestion of already-uploaded files."""

    filenames: list[str] = Field(
        ...,
        min_length=1,
        description="Files to ingest from the upload directory.",
    )
    chunking_strategy: Optional[ChunkingStrategyEnum] = None
    chunk_size: Optional[int] = Field(default=None, ge=50, le=8192)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=2048)
    background: bool = Field(
        default=False,
        description="If true, run ingestion in background and return job ID.",
    )


class QueryRequest(BaseModel):
    """Request to query the RAG system."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The user question.",
    )
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    retrieval_mode: RetrievalModeEnum = Field(default=RetrievalModeEnum.SIMILARITY)
    enable_reranking: Optional[bool] = None
    metadata_filter: Optional[dict[str, Any]] = None
    use_cache: bool = Field(default=True)


class ChatRequest(BaseModel):
    """Request for conversational chat with memory."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The user message.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Chat session ID. Generated if omitted.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for multi-user tracking.",
    )
    top_k: int = Field(default=5, ge=1, le=100)
    retrieval_mode: RetrievalModeEnum = Field(default=RetrievalModeEnum.SIMILARITY)
    enable_reranking: Optional[bool] = None
    stream: bool = Field(default=False)


class FeedbackRequest(BaseModel):
    """User feedback on a response."""

    query_id: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=2000)


# ── API Response Models ──────────────────────────────────────────────────────


class SourceDocument(BaseModel):
    """A source document returned alongside an answer."""

    content: str
    filename: str
    page_number: int
    chunk_id: str = ""
    document_id: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response from a query or chat request."""

    answer: str
    sources: list[SourceDocument]
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_level: str = Field(default="low", pattern="^(low|medium|high)$")
    model_used: str = ""
    tokens_used: int = 0
    cost_estimate: float = 0.0
    latency_ms: float = 0.0
    session_id: Optional[str] = None
    query_type: str = Field(default="unknown")
    validation: Optional[dict[str, Any]] = None


class StreamChunk(BaseModel):
    """A single chunk in a streaming response."""

    token: str
    query_id: str
    is_final: bool = False
    sources: Optional[list[SourceDocument]] = None
    metadata: Optional[dict[str, Any]] = None


class UploadResponse(BaseModel):
    """Response after uploading a document."""

    filename: str
    file_size: int
    message: str


class IngestResponse(BaseModel):
    """Response after ingestion completes."""

    documents: list[IngestedDocument]
    total_chunks: int
    message: str


class DocumentListResponse(BaseModel):
    """List of all ingested documents."""

    documents: list[IngestedDocument]
    total_count: int


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationHistory(BaseModel):
    """Full conversation history for a session."""

    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UsageStats(BaseModel):
    """Token usage and cost statistics."""

    total_queries: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    vector_store: str = ""
    embedding_model: str = ""
    llm_provider: str = ""
    documents_indexed: int = 0


class StatsResponse(BaseModel):
    """System statistics response."""

    total_documents: int = 0
    total_chunks: int = 0
    total_queries: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    vector_store_type: str = ""
    embedding_model: str = ""
    llm_provider: str = ""
    avg_latency_ms: float = 0.0


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: Optional[str] = None


# ── Standard Response Envelope ────────────────────────────────────────────


class PaginatedResponse(BaseModel):
    """Standard paginated response envelope."""

    data: list[Any] = Field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 50
    total_pages: int = 0


class ApiResponse(BaseModel):
    """Standard API response envelope."""

    data: Any = None
    message: Optional[str] = None
    success: bool = True


# ── Auth Schemas ───────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    """Login request payload."""

    email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    """User registration request."""

    email: str = Field(..., min_length=3, max_length=255)
    name: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    role: str = Field(default="employee", pattern="^(admin|employee|viewer)$")

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Require uppercase, lowercase, digit, and special character."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v


class ChangePasswordRequest(BaseModel):
    """Change password request."""

    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Require uppercase, lowercase, digit, and special character."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str


class UserResponse(BaseModel):
    """User information response."""

    id: str
    email: str
    name: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class UserListResponse(BaseModel):
    """List of users response."""

    users: list[UserResponse]
    total_count: int


class UpdateUserRequest(BaseModel):
    """Update user request (admin only)."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    role: Optional[str] = Field(None, pattern="^(admin|employee|viewer)$")
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8, max_length=128)
