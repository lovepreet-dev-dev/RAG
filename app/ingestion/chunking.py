"""Text chunking.

Recursive character splitting is the single supported strategy: it splits on
paragraph → sentence → word boundaries, which is fast, deterministic, and a
strong default for prose documents.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.constants import CHUNK_SEPARATORS_RECURSIVE
from app.config.settings import ChunkingStrategy, Settings, get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


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
    """Split documents into chunks.

    Args:
        documents: Raw documents to chunk.
        strategy: Chunking strategy to use.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        settings: Optional settings override.

    Returns:
        List of chunked Documents with updated metadata.

    Raises:
        ValueError: If the strategy is not supported.
    """
    if not documents:
        return []

    if settings is None:
        settings = get_settings()

    strategy = strategy or settings.chunking_strategy
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if strategy is not ChunkingStrategy.RECURSIVE:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    logger.info(
        "chunking_documents",
        strategy=strategy.value,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_count=len(documents),
    )

    chunks = _recursive_chunk(documents, chunk_size, chunk_overlap, settings)

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
