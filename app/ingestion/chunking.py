"""Text chunking strategies.

Provides three strategies — recursive, semantic, and sliding window — all
sharing a common ``chunk_documents`` entry point.  The strategy is selected
via the ``CHUNKING_STRATEGY`` environment variable.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, SemanticChunker

from app.config.constants import (
    CHUNK_SEPARATORS_RECURSIVE,
    CHUNK_SEPARATORS_SLIDING,
)
from app.config.settings import ChunkingStrategy, Settings, get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


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
    logger.info("chunking_complete", total_chunks=len(chunks))
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
    chunks: list[Document] = []
    for doc in documents:
        splits = splitter.split_documents([doc])
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
            split.metadata["total_chunks"] = len(splits)
            chunks.append(split)
    return chunks


def _semantic_chunk(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    settings: Settings,
) -> list[Document]:
    """Split using ``SemanticChunker`` based on embedding similarity.

    Groups sentences that are semantically similar into the same chunk.

    Args:
        documents: Documents to split.
        chunk_size: Max chunk size (used as fallback limit).
        chunk_overlap: Not used in semantic chunking.
        settings: Application settings.

    Returns:
        List of semantically chunked Documents.
    """
    from app.embeddings.factory import get_embeddings_model

    embeddings = get_embeddings_model(settings)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=settings.semantic_chunk_threshold,
    )
    chunks: list[Document] = []
    for doc in documents:
        splits = splitter.split_documents([doc])
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
            split.metadata["total_chunks"] = len(splits)
            chunks.append(split)
    return chunks


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
    chunks: list[Document] = []
    for doc in documents:
        splits = splitter.split_documents([doc])
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
            split.metadata["total_chunks"] = len(splits)
            chunks.append(split)
    return chunks
