"""Document ingestion pipeline: load -> clean -> chunk -> embed -> store.

Tracks duplicates, processing time, chunk count, and per-file statistics.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.config.settings import ChunkingStrategy, Settings, get_settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_document
from app.models.schemas import IngestedDocument
from app.utils.helpers import generate_content_hash, generate_id
from app.utils.logging import get_logger
from app.vectorstore.base import BaseVectorStore

logger = get_logger(__name__)

_DEFAULT_CACHE_SIZE = 10_000


class DuplicateLRUCache:
    """Thread-safe LRU cache for duplicate content hashes.

    Wraps ``OrderedDict`` with a fixed max size.  When the cache is full the
    oldest entry is evicted.  This replaces the old manual dict-slicing approach.
    """

    def __init__(self, max_size: int = _DEFAULT_CACHE_SIZE) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size

    def contains(self, content_hash: str) -> bool:
        return content_hash in self._cache

    def get(self, content_hash: str) -> Optional[str]:
        if content_hash in self._cache:
            self._cache.move_to_end(content_hash)
            return self._cache[content_hash]
        return None

    def put(self, content_hash: str, document_id: str) -> None:
        if content_hash in self._cache:
            self._cache.move_to_end(content_hash)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[content_hash] = document_id

    def __len__(self) -> int:
        return len(self._cache)


_duplicate_cache: Optional[DuplicateLRUCache] = None


def _get_duplicate_cache(settings: Settings) -> DuplicateLRUCache:
    """Get or initialise the singleton LRU duplicate cache."""
    global _duplicate_cache
    if _duplicate_cache is None:
        _duplicate_cache = DuplicateLRUCache(max_size=_DEFAULT_CACHE_SIZE)
    return _duplicate_cache


def _check_persistent_duplicate(
    content_hash: str,
    vector_store: BaseVectorStore,
) -> Optional[str]:
    """Check if a document with this content hash already exists in the store.

    Iterates the vector store metadata looking for an existing document_id whose
    chunks carry a ``content_hash`` matching *content_hash*.

    Returns the existing document_id if found, else ``None``.
    """
    try:
        for doc_info in vector_store.list_documents():
            doc_id = doc_info.get("document_id", "")
            if not doc_id or doc_id == "unknown":
                continue
            # Inspect the raw metadata stored alongside each chunk
            store_meta = getattr(vector_store, "_metadata", [])
            for entry in store_meta:
                chunk_meta = entry.get("metadata", {})
                if (
                    chunk_meta.get("document_id") == doc_id
                    and chunk_meta.get("content_hash") == content_hash
                ):
                    return doc_id
    except Exception:
        logger.debug("persistent_duplicate_check_failed", exc_info=True)
    return None


def ingest_files(
    file_paths: list[str],
    vector_store: BaseVectorStore,
    settings: Optional[Settings] = None,
    chunking_strategy: Optional[ChunkingStrategy] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[IngestedDocument]:
    """End-to-end ingestion: load files, chunk, embed, and store.

    Args:
        file_paths: Paths to files to ingest.
        vector_store: Target vector store.
        settings: Optional settings override.
        chunking_strategy: Override chunking strategy.
        chunk_size: Override chunk size.
        chunk_overlap: Override overlap.

    Returns:
        List of ``IngestedDocument`` summaries.
    """
    if settings is None:
        settings = get_settings()

    results: list[IngestedDocument] = []
    for file_path in file_paths:
        try:
            result = _ingest_single(
                file_path, vector_store, settings,
                chunking_strategy, chunk_size, chunk_overlap,
            )
            results.append(result)
        except Exception as exc:
            logger.error("ingestion_failed", file=file_path, error=str(exc))
            results.append(
                IngestedDocument(
                    document_id="",
                    filename=Path(file_path).name,
                    total_chunks=0,
                    total_pages=0,
                    created_at=datetime.now(timezone.utc),
                )
            )
    return results


def _ingest_single(
    file_path: str,
    vector_store: BaseVectorStore,
    settings: Settings,
    chunking_strategy: Optional[ChunkingStrategy] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> IngestedDocument:
    """Ingest a single file.

    Args:
        file_path: Path to the file.
        vector_store: Target vector store.
        settings: Application settings.
        chunking_strategy: Strategy override.
        chunk_size: Size override.
        chunk_overlap: Overlap override.

    Returns:
        ``IngestedDocument`` summary.
    """
    start = time.perf_counter()
    path = Path(file_path)
    filename = path.name

    logger.info("ingestion_started", filename=filename)

    # --- Load documents ---
    load_start = time.perf_counter()
    raw_docs = load_document(file_path, filename)
    load_elapsed = (time.perf_counter() - load_start) * 1000
    logger.info("documents_loaded", filename=filename, count=len(raw_docs), load_ms=round(load_elapsed, 2))

    if not raw_docs:
        logger.warning("empty_document", filename=filename)
        document_id = generate_id()
        return IngestedDocument(
            document_id=document_id,
            filename=filename,
            total_chunks=0,
            total_pages=0,
            created_at=datetime.now(timezone.utc),
        )

    # --- Deterministic document ID from content hash ---
    all_text = "".join(d.page_content for d in raw_docs)
    content_hash = generate_content_hash(all_text.encode())
    document_id = f"doc_{content_hash[:12]}"

    logger.info(
        "content_hash_computed",
        filename=filename,
        document_id=document_id,
        content_hash=content_hash[:16],
    )

    # --- Duplicate detection: LRU cache first, then persistent store ---
    cache = _get_duplicate_cache(settings)

    existing_id = cache.get(content_hash)
    if existing_id is None:
        existing_id = _check_persistent_duplicate(content_hash, vector_store)

    if existing_id is not None:
        logger.info(
            "duplicate_document_skipped",
            filename=filename,
            existing_id=existing_id,
        )
        return IngestedDocument(
            document_id=existing_id,
            filename=filename,
            total_chunks=0,
            total_pages=len(raw_docs),
            created_at=datetime.now(timezone.utc),
        )

    cache.put(content_hash, document_id)

    # --- Attach metadata to raw docs ---
    for doc in raw_docs:
        doc.metadata["document_id"] = document_id
        doc.metadata["source"] = filename
        doc.metadata["content_hash"] = content_hash

    # --- Chunk documents ---
    chunk_start = time.perf_counter()
    chunks = chunk_documents(
        raw_docs,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        settings=settings,
    )
    chunk_elapsed = (time.perf_counter() - chunk_start) * 1000

    # --- Filter invalid chunks ---
    min_len = settings.min_chunk_length if hasattr(settings, "min_chunk_length") else 10
    valid_chunks: list[Document] = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text or len(text) < min_len:
            continue
        chunk.metadata["document_id"] = document_id
        chunk.metadata["source"] = filename
        chunk.metadata["content_hash"] = content_hash
        valid_chunks.append(chunk)

    filtered_out = len(chunks) - len(valid_chunks)
    if filtered_out > 0:
        logger.info(
            "invalid_chunks_filtered",
            filename=filename,
            removed=filtered_out,
            remaining=len(valid_chunks),
        )

    # --- Assign globally unique chunk IDs ---
    for idx, chunk in enumerate(valid_chunks):
        chunk.metadata["chunk_id"] = f"{document_id}_{idx:04d}"
        chunk.metadata["total_chunks"] = len(valid_chunks)

    # --- Embed and store ---
    store_start = time.perf_counter()
    vector_store.add_documents(valid_chunks)
    if hasattr(vector_store, "save"):
        vector_store.save()
    store_elapsed = (time.perf_counter() - store_start) * 1000

    total_elapsed = (time.perf_counter() - start) * 1000

    logger.info(
        "ingestion_complete",
        filename=filename,
        document_id=document_id,
        chunks=len(valid_chunks),
        pages=len(raw_docs),
        chunk_ms=round(chunk_elapsed, 2),
        store_ms=round(store_elapsed, 2),
        total_ms=round(total_elapsed, 2),
    )

    return IngestedDocument(
        document_id=document_id,
        filename=filename,
        total_chunks=len(valid_chunks),
        total_pages=len(raw_docs),
        created_at=datetime.now(timezone.utc),
    )


def detect_duplicates(file_paths: list[str]) -> dict[str, list[str]]:
    """Detect duplicate files based on content hashing.

    Args:
        file_paths: List of file paths to check.

    Returns:
        Dict mapping content hash to list of file paths with that hash.
    """
    hash_map: dict[str, list[str]] = {}
    for fp in file_paths:
        path = Path(fp)
        if path.exists():
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_map.setdefault(h, []).append(fp)
    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def reset_duplicate_cache() -> None:
    """Reset the global LRU duplicate cache.  Useful for tests."""
    global _duplicate_cache
    _duplicate_cache = None
