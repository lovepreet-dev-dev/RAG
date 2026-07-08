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

import warnings
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
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LMSTUDIO = "lmstudio"
    OPENROUTER = "openrouter"


class EmbeddingModelType(str, Enum):
    """Enumeration of supported embedding model families.

    The actual HuggingFace model name is resolved in
    ``app.config.constants.EMBEDDING_MODEL_MAP``.
    """

    BGE_SMALL = "bge-small"
    BGE_LARGE = "bge-large"
    MINILM = "minilm"
    OPENAI = "openai"
    NOMIC = "nomic"


class VectorStoreType(str, Enum):
    """Enumeration of supported vector store backends.

    Switching the backend requires only setting ``VECTOR_STORE``
    in the environment — no code changes needed.
    """

    FAISS = "faiss"
    CHROMA = "chroma"
    QDRANT = "qdrant"
    PINECONE = "pinecone"


class ChunkingStrategy(str, Enum):
    """Text chunking strategies available in the ingestion pipeline.

    ``recursive``  — fast, rule-based, good default.
    ``semantic``   — groups sentences by embedding similarity.
    ``sliding_window`` — fixed-size windows with overlap.
    """

    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"


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

    # Google Gemini
    gemini_api_key: str = Field(default="", description="Google AI / Gemini API key.")
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model identifier.",
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

    # LM Studio (local)
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Base URL of the LM Studio OpenAI-compatible server.",
    )
    lmstudio_model: str = Field(
        default="local-model",
        description="Model name reported by LM Studio.",
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

    # Chroma-specific
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="Chroma persistence directory.",
    )
    chroma_collection: str = Field(
        default="rag_documents",
        description="Chroma collection name.",
    )

    # Qdrant-specific
    qdrant_host: str = Field(default="localhost", description="Qdrant host.")
    qdrant_port: int = Field(default=6333, description="Qdrant port.")
    qdrant_collection: str = Field(
        default="rag_documents",
        description="Qdrant collection name.",
    )

    # Pinecone-specific
    pinecone_api_key: str = Field(default="", description="Pinecone API key.")
    pinecone_index: str = Field(
        default="rag-documents",
        description="Pinecone index name.",
    )
    pinecone_environment: str = Field(
        default="us-east-1",
        description="Pinecone cloud region.",
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
    semantic_chunk_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Percentile breakpoint for semantic chunking.",
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
    mmr_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Top-K results from MMR search.",
    )
    hybrid_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Top-K results from hybrid search.",
    )

    # Strategy toggles
    enable_mmr: bool = Field(
        default=True,
        description="Enable Maximal Marginal Relevance for diversity.",
    )
    enable_hybrid_search: bool = Field(
        default=True,
        description="Enable hybrid semantic + keyword search.",
    )
    enable_multi_query: bool = Field(
        default=True,
        description="Enable multi-query fallback when confidence is low.",
    )

    # MMR settings
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR diversity weight (0=max diversity, 1=max relevance).",
    )

    # Hybrid settings
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for semantic score in hybrid mode.",
    )

    # Multi-query fallback
    multi_query_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold below which multi-query fallback triggers.",
    )
    multi_query_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of alternative queries to generate in multi-query fallback.",
    )

    # Context limits
    max_context_tokens: int = Field(
        default=3000,
        ge=100,
        le=128000,
        description="Maximum total tokens for context sent to the LLM.",
    )

    # Query preprocessing
    enable_query_preprocessing: bool = Field(
        default=True,
        description="Enable query normalization and metadata extraction.",
    )
    enable_query_classification: bool = Field(
        default=True,
        description="Enable query type classification for strategy routing.",
    )

    # Query expansion
    enable_hyde: bool = Field(
        default=False,
        description="Enable HyDE (Hypothetical Document Embeddings) expansion.",
    )
    enable_synonym_expansion: bool = Field(
        default=True,
        description="Enable synonym-based query expansion.",
    )
    enable_keyword_expansion: bool = Field(
        default=True,
        description="Enable keyword extraction and expansion.",
    )

    # Answer validation
    enable_answer_validation: bool = Field(
        default=True,
        description="Enable post-generation answer validation.",
    )

    # Parent document retrieval
    enable_parent_document: bool = Field(
        default=False,
        description="Retrieve parent documents instead of child chunks.",
    )

    # Metadata search
    enable_metadata_search: bool = Field(
        default=True,
        description="Enable metadata-aware search alongside semantic search.",
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

    # ── Data Paths ───────────────────────────────────────────────────────

    data_dir: str = Field(
        default="./data",
        description="Root data directory for uploads, indexes, and temp files.",
    )

    # ── Cost Tracking ────────────────────────────────────────────────────

    openai_cost_per_1k_input_tokens: float = Field(
        default=0.005,
        ge=0.0,
        description="Cost per 1k input tokens (USD).",
    )
    openai_cost_per_1k_output_tokens: float = Field(
        default=0.015,
        ge=0.0,
        description="Cost per 1k output tokens (USD).",
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
        """Warn if auth is enabled but the secret key is the default placeholder.

        Returns:
            The validated Settings instance.
        """
        if self.enable_auth and self.secret_key == "change-this-to-a-random-secret-key":
            warnings.warn(
                "Authentication is enabled but SECRET_KEY is the default placeholder. "
                "Generate a secure key for production use.",
                stacklevel=1,
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
            LLMProvider.GEMINI: self.gemini_model,
            LLMProvider.LMSTUDIO: self.lmstudio_model,
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
            LLMProvider.GEMINI: self.gemini_api_key,
            LLMProvider.LMSTUDIO: "",
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
            LLMProvider.GEMINI: None,
            LLMProvider.LMSTUDIO: self.lmstudio_base_url,
            LLMProvider.OLLAMA: self.ollama_base_url,
            LLMProvider.OPENROUTER: self.openrouter_base_url,
        }
        return url_map[self.llm_provider]

    def requires_api_key(self) -> bool:
        """Check whether the configured LLM provider needs an API key.

        Returns:
            ``True`` if an API key is required.
        """
        return self.llm_provider not in {LLMProvider.OLLAMA, LLMProvider.LMSTUDIO}

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

        if self.vector_store == VectorStoreType.PINECONE and not self.pinecone_api_key:
            issues.append("PINECONE_API_KEY is not set.")

        if self.embedding_model == EmbeddingModelType.OPENAI and not self.openai_api_key:
            issues.append("Embedding model is OpenAI but OPENAI_API_KEY is not set.")

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
