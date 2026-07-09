"""Text chunking strategies.

Provides three strategies — recursive, semantic, and sliding window — all
sharing a common ``chunk_documents`` entry point.  The strategy is selected
via the ``CHUNKING_STRATEGY`` environment variable.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.constants import (
    CHUNK_SEPARATORS_RECURSIVE,
    CHUNK_SEPARATORS_SLIDING,
)
from app.config.settings import ChunkingStrategy, Settings, get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# ── SemanticChunker cache ──────────────────────────────────────────────
# Avoids reconstructing the splitter on every call.
_semantic_chunker_cache: dict[str, SemanticChunker] = {}


def _get_semantic_chunker(settings: Settings) -> SemanticChunker:
    """Return a cached SemanticChunker instance for the given settings."""
    from app.embeddings.factory import get_embeddings_model

    cache_key = f"{settings.embedding_model}_{settings.semantic_chunk_threshold}"
    if cache_key not in _semantic_chunker_cache:
        embeddings = get_embeddings_model(settings)
        _semantic_chunker_cache[cache_key] = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=settings.semantic_chunk_threshold,
        )
    return _semantic_chunker_cache[cache_key]


def _is_valid_chunk(text: str, min_length: int = 10) -> bool:
    """Return True if the chunk text is non-empty and meets min_length."""
    stripped = text.strip()
    return bool(stripped) and len(stripped) >= min_length


def chunk_documents(
    documents: list[Document],
    strategy: Optional[ChunkingStrategy] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> list[Document]:
    """Split documents into chunks using the specified strategy.

    Args:
        documents: Raw documents to chunk.
        strategy: Chunking strategy to use.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        settings: Optional settings override.

    Returns:
        List of chunked Documents with updated metadata.

    Raises:
        ValueError: If no documents are provided.
    """
    if not documents:
        return []

    if settings is None:
        settings = get_settings()

    strategy = strategy or settings.chunking_strategy
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    logger.info(
        "chunking_documents",
        strategy=strategy.value,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_count=len(documents),
    )

    chunkers = {
        ChunkingStrategy.RECURSIVE: _recursive_chunk,
        ChunkingStrategy.SEMANTIC: _semantic_chunk,
        ChunkingStrategy.SLIDING_WINDOW: _sliding_window_chunk,
    }

    chunker_fn = chunkers.get(strategy)
    if chunker_fn is None:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    chunks = chunker_fn(documents, chunk_size, chunk_overlap, settings)

    # --- Filter invalid chunks (empty, whitespace-only, too short) ---
    min_len = settings.min_chunk_length if hasattr(settings, "min_chunk_length") else 10
    valid_chunks: list[Document] = []
    for chunk in chunks:
        if _is_valid_chunk(chunk.page_content, min_len):
            valid_chunks.append(chunk)

    filtered = len(chunks) - len(valid_chunks)
    if filtered > 0:
        logger.info(
            "chunks_filtered",
            total=len(chunks),
            valid=len(valid_chunks),
            removed=filtered,
        )

    logger.info("chunking_complete", total_chunks=len(valid_chunks))
    return valid_chunks


def _assign_global_ids(chunks: list[Document], doc_id: str) -> list[Document]:
    """Assign globally unique chunk IDs using document_id + sequential index."""
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{doc_id}_{idx:04d}"
        chunk.metadata["total_chunks"] = len(chunks)
    return chunks


def _recursive_chunk(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    settings: Settings,
) -> list[Document]:
    """Split using ``RecursiveCharacterTextSplitter``.

    Tries to split on paragraph, sentence, word boundaries in order.

    Args:
        documents: Documents to split.
        chunk_size: Max chunk size.
        chunk_overlap: Overlap size.
        settings: Application settings.

    Returns:
        List of chunked Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=CHUNK_SEPARATORS_RECURSIVE,
        keep_separator=True,
    )
    all_chunks: list[Document] = []
    for doc in documents:
        doc_id = doc.metadata.get("document_id", "unknown")
        splits = splitter.split_documents([doc])
        # Assign globally unique IDs per document
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = f"{doc_id}_{i:04d}"
            split.metadata["total_chunks"] = len(splits)
        all_chunks.extend(splits)
    return all_chunks


def _semantic_chunk(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    settings: Settings,
) -> list[Document]:
    """Split using ``SemanticChunker`` based on embedding similarity.

    Groups sentences that are semantically similar into the same chunk.
    Uses a cached SemanticChunker instance.

    Args:
        documents: Documents to split.
        chunk_size: Max chunk size (used as fallback limit).
        chunk_overlap: Not used in semantic chunking.
        settings: Application settings.

    Returns:
        List of semantically chunked Documents.
    """
    splitter = _get_semantic_chunker(settings)
    all_chunks: list[Document] = []
    for doc in documents:
        doc_id = doc.metadata.get("document_id", "unknown")
        splits = splitter.split_documents([doc])
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = f"{doc_id}_{i:04d}"
            split.metadata["total_chunks"] = len(splits)
        all_chunks.extend(splits)
    return all_chunks


def _sliding_window_chunk(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    settings: Settings,
) -> list[Document]:
    """Split using a fixed-size sliding window.

    Args:
        documents: Documents to split.
        chunk_size: Window size in characters.
        chunk_overlap: Overlap between windows.
        settings: Application settings.

    Returns:
        List of windowed Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=CHUNK_SEPARATORS_SLIDING,
        keep_separator=True,
    )
    all_chunks: list[Document] = []
    for doc in documents:
        doc_id = doc.metadata.get("document_id", "unknown")
        splits = splitter.split_documents([doc])
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = f"{doc_id}_{i:04d}"
            split.metadata["total_chunks"] = len(splits)
        all_chunks.extend(splits)
    return all_chunks


def reset_semantic_chunker_cache() -> None:
    """Reset the SemanticChunker cache.  Useful for tests."""
    global _semantic_chunker_cache
    _semantic_chunker_cache.clear()
