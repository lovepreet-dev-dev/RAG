"""Document ingestion pipeline: load → clean → chunk → embed → store.

Tracks duplicates, processing time, chunk count, and per-file statistics.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.config.settings import ChunkingStrategy, Settings, get_settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_document
from app.models.schemas import IngestedDocument
from app.utils.helpers import generate_id
from app.utils.logging import get_logger
from app.vectorstore.base import BaseVectorStore

logger = get_logger(__name__)

_duplicate_hashes: dict[str, str] = {}
_MAX_DUPLICATE_HASHES = 10_000


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
            raise
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
    document_id = generate_id()

    logger.info("starting_ingestion", filename=filename, document_id=document_id)

    raw_docs = load_document(file_path, filename)
    for doc in raw_docs:
        doc.metadata["document_id"] = document_id

    content_hash = hashlib.sha256(
        "".join(d.page_content for d in raw_docs).encode()
    ).hexdigest()

    now = datetime.now(timezone.utc)

    if content_hash in _duplicate_hashes:
        existing_id = _duplicate_hashes[content_hash]
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
            created_at=now,
        )

    _duplicate_hashes[content_hash] = document_id
    if len(_duplicate_hashes) > _MAX_DUPLICATE_HASHES:
        oldest = list(_duplicate_hashes.keys())[:_MAX_DUPLICATE_HASHES // 2]
        for k in oldest:
            _duplicate_hashes.pop(k, None)

    chunks = chunk_documents(
        raw_docs,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        settings=settings,
    )

    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["source"] = filename

    vector_store.add_documents(chunks)

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "ingestion_complete",
        filename=filename,
        document_id=document_id,
        chunks=len(chunks),
        pages=len(raw_docs),
        latency_ms=round(elapsed, 2),
    )

    return IngestedDocument(
        document_id=document_id,
        filename=filename,
        total_chunks=len(chunks),
        total_pages=len(raw_docs),
        created_at=now,
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
            content = path.read_bytes()
            h = hashlib.sha256(content).hexdigest()
            hash_map.setdefault(h, []).append(fp)
    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}
