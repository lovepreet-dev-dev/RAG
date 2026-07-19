"""Centralized constants for the RAG platform.

All prompt templates, error messages, supported extensions, model mappings,
magic numbers, and default values live here.  Keeping them in one place
prevents drift and makes prompt engineering a single-file edit.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
TEST_DATA_DIR = BASE_DIR / "tests" / "fixtures"
UPLOAD_DIR = DATA_DIR / "uploads"

# ── Supported file extensions ────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "markdown",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
}

SUPPORTED_MIME_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "text": "text/plain",
    "markdown": "text/markdown",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "html": "text/html",
    "csv": "text/csv",
    "json": "application/json",
    "jsonl": "application/jsonl",
}

# ── Embedding model map ──────────────────────────────────────────────────────

EMBEDDING_MODEL_MAP: dict[str, str] = {
    "bge-small": "BAAI/bge-small-en-v1.5",
}

EMBEDDING_DIMENSIONS: dict[str, int] = {
    "bge-small": 384,
}

# ── Chunking defaults ────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128
MIN_CHUNK_SIZE = 50
MAX_CHUNK_SIZE = 8192
CHUNK_SEPARATORS_RECURSIVE: list[str] = ["\n\n", "\n", ". ", " ", ""]

# ── Retrieval defaults ───────────────────────────────────────────────────────

DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.3
MAX_TOP_K = 100
DEFAULT_HYBRID_ALPHA = 0.5

# ── API ──────────────────────────────────────────────────────────────────────

API_V1_PREFIX = "/api/v1"
DOCS_URL = "/docs"
REDOCS_URL = "/redoc"
APP_TITLE = "RAG Engine"
APP_DESCRIPTION = "Production-ready Retrieval-Augmented Generation API"
APP_VERSION = "1.0.0"

# ── Error messages ───────────────────────────────────────────────────────────

ERR_UNSUPPORTED_FILE = "Unsupported file type. Supported: PDF, TXT, DOCX, MD, HTML, CSV, JSON"
ERR_NO_FILENAME = "No filename provided."
ERR_FILE_NOT_FOUND = "File not found: {path}"
ERR_INGESTION_FAILED = "Ingestion failed: {error}"
ERR_QUERY_FAILED = "Query failed: {error}"
ERR_CHAT_FAILED = "Chat failed: {error}"
ERR_DELETE_FAILED = "Delete failed: {error}"
ERR_LIST_FAILED = "Failed to list documents: {error}"
ERR_UPLOAD_FAILED = "Failed to save file: {error}"
ERR_DOCUMENT_NOT_FOUND = "Document '{doc_id}' not found or could not be deleted."
ERR_INTERNAL_SERVER = "Internal server error. Please try again later."
ERR_LLM_NOT_CONFIGURED = "LLM provider '{provider}' is not properly configured."
ERR_VECTOR_STORE_NOT_AVAILABLE = "Vector store '{store}' is not available."

# ── Prompt Templates ─────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a helpful, accurate, and truthful AI assistant.

RULES:
1. Answer ONLY based on the provided context. Never fabricate information.
2. If the context does not contain enough information, say "I don't have enough information to answer that question."
3. Always cite your sources by referencing the document name and page number.
4. Explain your reasoning clearly and step by step when appropriate.
5. If multiple sources provide different perspectives, present them fairly.
6. Never make up facts, statistics, or references.
7. Be concise but thorough.
8. When unsure, acknowledge the uncertainty rather than guessing."""

RAG_USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above. Include citations to the source documents."""

CONDENSE_QUESTION_PROMPT = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question that can be understood
without the conversation history.

Chat History:
{chat_history}

Follow Up Question: {question}

Standalone question:"""

ANSWER_VALIDATION_PROMPT = """You are a fact-checking assistant. Given a question, answer, and source documents,
determine which statements in the answer are NOT supported by the sources.

Question: {query}

Answer to check:
{answer}

Source documents:
{context}

List each unsupported statement on a new line. If all statements are supported, reply with "NONE".
Be strict — only mark as unsupported if the information genuinely cannot be found in the sources."""

# ── Logging format ───────────────────────────────────────────────────────────

LOG_FORMAT_JSON = True
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# ── Rate limiting ────────────────────────────────────────────────────────────

DEFAULT_RATE_LIMIT = 60
DEFAULT_RATE_WINDOW = 60
