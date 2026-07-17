"""Central configuration module for the RAG platform.

Loads all settings from environment variables or a ``.env`` file using
``pydantic-settings``.  Every setting has a safe default so the application
can boot with zero configuration.  Switching any backend (LLM, embeddings,
vector store) is a single environment-variable change.

Design decisions
----------------
* ``BaseSettings`` (pydantic v2) gives automatic type coercion, validation,
  and ``.env`` file loading with zero boilerplate.
* Enums constrain allowed values at **startup** time, catching mis-configuration
  before the first request arrives.
* ``lru_cache(maxsize=1)`` on ``get_settings()`` ensures the ``.env`` file is
  parsed exactly once per process — safe for threads and cheap for imports.
* No directory-creation side-effects in validators; callers decide when to
  create directories.  This keeps ``Settings`` pure and testable.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ────────────────────────────────────────────────────────────────────────────
# Enums
# ────────────────────────────────────────────────────────────────────────────


class LLMProvider(str, Enum):
    """Enumeration of supported Large Language Model providers.

    Each value maps to a factory function in ``app.llm.factory``.
    OpenRouter proxies most other model families (Gemini, Llama, Mistral, …)
    through an OpenAI-compatible API, and any OpenAI-compatible local server
    (e.g. LM Studio) works via ``OPENAI_BASE_URL``.
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class EmbeddingModelType(str, Enum):
    """Enumeration of supported embedding model families.

    The actual HuggingFace model name is resolved in
    ``app.config.constants.EMBEDDING_MODEL_MAP``.
    """

    BGE_SMALL = "bge-small"


class VectorStoreType(str, Enum):
    """Enumeration of supported vector store backends."""

    FAISS = "faiss"


class ChunkingStrategy(str, Enum):
    """Text chunking strategies available in the ingestion pipeline.

    ``recursive`` — splits on paragraph → sentence → word boundaries.
    """

    RECURSIVE = "recursive"


# ────────────────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables / ``.env``.

    All fields provide safe defaults so the application boots with zero
    configuration.  Override any value via env vars or the ``.env`` file.
    ``extra="ignore"`` prevents crashes when unrelated env vars are present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ─────────────────────────────────────────────────────

    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Which LLM backend to use for generation.",
    )

    # OpenAI / compatible
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the OpenAI-compatible API.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Model name for OpenAI completions.",
    )

    # Anthropic
    anthropic_api_key: str = Field(default="", description="Anthropic API key.")
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Anthropic model identifier.",
    )

    # Ollama (local)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama server.",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model tag to use.",
    )

    # OpenRouter
    openrouter_api_key: str = Field(default="", description="OpenRouter API key.")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL.",
    )
    openrouter_model: str = Field(
        default="meta-llama/llama-3.1-8b-instruct:free",
        description="OpenRouter model identifier.",
    )

    # ── Embeddings ───────────────────────────────────────────────────────

    embedding_model: EmbeddingModelType = Field(
        default=EmbeddingModelType.BGE_SMALL,
        description="Embedding model family to use.",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for sentence-transformer models ('cpu' or 'cuda').",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name (when EMBEDDING_MODEL=openai).",
    )

    # ── Vector Store ─────────────────────────────────────────────────────

    vector_store: VectorStoreType = Field(
        default=VectorStoreType.FAISS,
        description="Vector store backend.",
    )

    # FAISS-specific
    faiss_index_dir: str = Field(
        default="./data/faiss_index",
        description="Directory for persisting the FAISS index.",
    )

    # ── Chunking ─────────────────────────────────────────────────────────

    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.RECURSIVE,
        description="Text splitting strategy.",
    )
    chunk_size: int = Field(
        default=512,
        ge=50,
        le=8192,
        description="Maximum chunk size in characters.",
    )
    chunk_overlap: int = Field(
        default=128,
        ge=0,
        le=2048,
        description="Overlap between consecutive chunks.",
    )
    min_chunk_length: int = Field(
        default=10,
        ge=0,
        description="Minimum character length for chunks to be kept.",
    )

    # ── Ingestion ────────────────────────────────────────────────────────

    enable_ocr: bool = Field(
        default=False,
        description="Enable OCR fallback for PDFs with no extractable text.",
    )
    ocr_language: str = Field(
        default="eng",
        description="Tesseract OCR language code (e.g. 'eng', 'fra').",
    )
    csv_batch_size: int = Field(
        default=500,
        ge=10,
        le=5000,
        description="Number of rows per batch when ingesting large CSV files.",
    )
    json_batch_size: int = Field(
        default=500,
        ge=10,
        le=5000,
        description="Number of items per batch when ingesting large JSON arrays.",
    )

    # ── Retrieval ────────────────────────────────────────────────────────

    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of documents to retrieve (legacy alias for similarity_top_k).",
    )
    score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to include a result.",
    )

    # Per-strategy Top-K
    similarity_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Top-K results from similarity search.",
    )
    hybrid_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Top-K results from hybrid search.",
    )

    # Strategy toggles
    enable_hybrid_search: bool = Field(
        default=True,
        description="Enable hybrid semantic + keyword search.",
    )

    # Hybrid settings
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for semantic score in hybrid mode.",
    )

    # Context limits
    max_context_tokens: int = Field(
        default=3000,
        ge=100,
        le=128000,
        description="Maximum total tokens for context sent to the LLM.",
    )

    # Answer validation
    enable_answer_validation: bool = Field(
        default=True,
        description="Enable post-generation answer validation.",
    )

    # ── Reranking ────────────────────────────────────────────────────────

    enable_reranking: bool = Field(
        default=False,
        description="Re-score retrieved chunks with a cross-encoder.",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="HuggingFace cross-encoder model for reranking.",
    )
    rerank_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to keep after reranking.",
    )

    # ── Chat / Memory ────────────────────────────────────────────────────

    enable_memory: bool = Field(
        default=True,
        description="Maintain per-session conversation history.",
    )
    memory_window: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recent message pairs to keep in memory.",
    )
    session_timeout_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours before an inactive session is purged.",
    )

    # ── Server ───────────────────────────────────────────────────────────

    host: str = Field(default="0.0.0.0", description="Bind address.")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port.")
    workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of uvicorn worker processes.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    debug: bool = Field(default=False, description="Enable debug mode.")

    # ── Timeouts & Retries ───────────────────────────────────────────────

    llm_timeout_seconds: int = Field(
        default=120,
        ge=5,
        le=600,
        description="Timeout in seconds for LLM API calls.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries for failed LLM calls.",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding computation.",
    )

    # ── Uploads ──────────────────────────────────────────────────────────

    upload_dir: str = Field(
        default="./data/uploads",
        description="Directory for uploaded files.",
    )
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum upload file size in megabytes.",
    )

    # ── Rate Limiting ────────────────────────────────────────────────────

    rate_limit_requests: int = Field(
        default=60,
        ge=1,
        description="Max requests per window per client.",
    )
    rate_limit_window: int = Field(
        default=60,
        ge=1,
        description="Rate limit window in seconds.",
    )

    # ── CORS ───────────────────────────────────────────────────────────

    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins. Use ['*'] for open access.",
    )

    # ── Authentication ───────────────────────────────────────────────────

    enable_auth: bool = Field(
        default=False,
        description="Require JWT bearer tokens for API access.",
    )
    secret_key: str = Field(
        default="change-this-to-a-random-secret-key",
        description="Signing key for JWT tokens.",
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm.",
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="JWT token lifetime in minutes.",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Refresh token lifetime in days.",
    )

    # ── Database ─────────────────────────────────────────────────────────

    database_url: str = Field(
        default="sqlite:///./data/rag.db",
        description="SQLAlchemy database URL (SQLite or PostgreSQL).",
    )

    # ── Data Paths ───────────────────────────────────────────────────────

    data_dir: str = Field(
        default="./data",
        description="Root data directory for uploads, indexes, and temp files.",
    )

    # ── Cost Tracking ────────────────────────────────────────────────────
    # Per-provider pricing so cost estimates are honest for every backend.
    # Local providers (Ollama) always cost 0. OpenRouter pricing varies per
    # model, so it defaults to 0 unless configured.

    openai_cost_per_1k_input_tokens: float = Field(
        default=0.005,
        ge=0.0,
        description="OpenAI cost per 1k input tokens (USD).",
    )
    openai_cost_per_1k_output_tokens: float = Field(
        default=0.015,
        ge=0.0,
        description="OpenAI cost per 1k output tokens (USD).",
    )
    anthropic_cost_per_1k_input_tokens: float = Field(
        default=0.003,
        ge=0.0,
        description="Anthropic cost per 1k input tokens (USD).",
    )
    anthropic_cost_per_1k_output_tokens: float = Field(
        default=0.015,
        ge=0.0,
        description="Anthropic cost per 1k output tokens (USD).",
    )
    openrouter_cost_per_1k_input_tokens: float = Field(
        default=0.0,
        ge=0.0,
        description="OpenRouter cost per 1k input tokens (USD); model-dependent.",
    )
    openrouter_cost_per_1k_output_tokens: float = Field(
        default=0.0,
        ge=0.0,
        description="OpenRouter cost per 1k output tokens (USD); model-dependent.",
    )

    # ──────────────────────────────────────────────────────────────────────
    # Validators
    # ──────────────────────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure the log level is a recognised Python logging level.

        Args:
            v: Raw log level string.

        Returns:
            Upper-cased log level string.

        Raises:
            ValueError: If the string is not a valid log level.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            msg = f"Invalid log level '{v}'. Must be one of: {sorted(valid_levels)}"
            raise ValueError(msg)
        return upper

    @field_validator("embedding_device")
    @classmethod
    def validate_embedding_device(cls, v: str) -> str:
        """Ensure the device string is valid.

        Args:
            v: Raw device string.

        Returns:
            Lower-cased device string.

        Raises:
            ValueError: If the device is not recognised.
        """
        valid_devices = {"cpu", "cuda", "mps"}
        lower = v.lower()
        if lower not in valid_devices:
            msg = f"Invalid embedding device '{v}'. Must be one of: {sorted(valid_devices)}"
            raise ValueError(msg)
        return lower

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> Settings:
        """Chunk overlap must be strictly less than chunk size.

        Returns:
            The validated Settings instance.

        Raises:
            ValueError: If overlap >= chunk_size.
        """
        if self.chunk_overlap >= self.chunk_size:
            msg = (
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_auth_secret(self) -> Settings:
        """Require a non-default secret key when auth is enabled.

        Returns:
            The validated Settings instance.

        Raises:
            ValueError: If auth is enabled but SECRET_KEY is the default placeholder.
        """
        if self.enable_auth and self.secret_key == "change-this-to-a-random-secret-key":
            raise ValueError(
                "SECRET_KEY must be changed from the default placeholder when "
                "authentication is enabled. Generate a secure key with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(64))\""
            )
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────────────────────────────────

    def get_llm_model_name(self) -> str:
        """Return the model name string for the currently configured provider.

        Returns:
            Model name (e.g. ``"gpt-4o-mini"``).
        """
        model_map: dict[LLMProvider, str] = {
            LLMProvider.OPENAI: self.openai_model,
            LLMProvider.OLLAMA: self.ollama_model,
            LLMProvider.ANTHROPIC: self.anthropic_model,
            LLMProvider.OPENROUTER: self.openrouter_model,
        }
        return model_map[self.llm_provider]

    def get_llm_api_key(self) -> str:
        """Return the API key for the currently configured provider.

        Returns:
            API key string (empty string for local providers).
        """
        key_map: dict[LLMProvider, str] = {
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OLLAMA: "",
            LLMProvider.OPENROUTER: self.openrouter_api_key,
        }
        return key_map[self.llm_provider]

    def get_llm_base_url(self) -> Optional[str]:
        """Return the base URL for the currently configured provider.

        Returns:
            Base URL string, or ``None`` for providers that don't use one.
        """
        url_map: dict[LLMProvider, Optional[str]] = {
            LLMProvider.OPENAI: self.openai_base_url,
            LLMProvider.ANTHROPIC: None,
            LLMProvider.OLLAMA: self.ollama_base_url,
            LLMProvider.OPENROUTER: self.openrouter_base_url,
        }
        return url_map[self.llm_provider]

    def requires_api_key(self) -> bool:
        """Check whether the configured LLM provider needs an API key.

        Returns:
            ``True`` if an API key is required.
        """
        return self.llm_provider is not LLMProvider.OLLAMA

    def get_cost_per_1k_tokens(self) -> tuple[float, float]:
        """Return (input, output) cost per 1k tokens for the active provider.

        Returns:
            Tuple of USD costs; ``(0.0, 0.0)`` for local providers.
        """
        cost_map: dict[LLMProvider, tuple[float, float]] = {
            LLMProvider.OPENAI: (
                self.openai_cost_per_1k_input_tokens,
                self.openai_cost_per_1k_output_tokens,
            ),
            LLMProvider.ANTHROPIC: (
                self.anthropic_cost_per_1k_input_tokens,
                self.anthropic_cost_per_1k_output_tokens,
            ),
            LLMProvider.OPENROUTER: (
                self.openrouter_cost_per_1k_input_tokens,
                self.openrouter_cost_per_1k_output_tokens,
            ),
            LLMProvider.OLLAMA: (0.0, 0.0),
        }
        return cost_map[self.llm_provider]

    def max_upload_bytes(self) -> int:
        """Return the maximum upload size in bytes.

        Returns:
            Maximum upload size in bytes.
        """
        return self.max_upload_size_mb * 1024 * 1024

    def validate_for_production(self) -> list[str]:
        """Run a suite of production-readiness checks.

        Returns:
            List of warning messages.  Empty list means all checks passed.
        """
        issues: list[str] = []

        if self.debug:
            issues.append("DEBUG mode is enabled — disable it in production.")

        if self.secret_key == "change-this-to-a-random-secret-key":
            issues.append("SECRET_KEY is the default placeholder — change it.")

        if self.requires_api_key() and not self.get_llm_api_key():
            issues.append(
                f"LLM provider '{self.llm_provider.value}' requires an API key "
                f"but none was provided."
            )

        if self.log_level == "DEBUG":
            issues.append("Log level is DEBUG — expect high verbosity in production.")

        return issues


# ────────────────────────────────────────────────────────────────────────────
# Singleton
# ────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance.

    ``lru_cache(maxsize=1)`` ensures the ``.env`` file is parsed exactly
    once, regardless of how many modules import this function.

    Returns:
        Cached ``Settings`` instance.
    """
    return Settings()
