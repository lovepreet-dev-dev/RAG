"""Tests for text chunking strategies."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.config.settings import ChunkingStrategy, Settings
from app.ingestion.chunking import chunk_documents


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content=(
                "Machine learning is a subset of artificial intelligence. "
                "It enables systems to learn from data. "
                "Deep learning is a subset of machine learning. "
                "Neural networks are the foundation of deep learning. "
                "Transformers have revolutionized natural language processing. "
                "BERT and GPT are popular transformer models. "
                "Retrieval-Augmented Generation combines retrieval with generation. "
                "RAG systems retrieve relevant documents before generating answers. "
                "Vector databases store document embeddings for fast retrieval. "
                "FAISS, Chroma, and Qdrant are popular vector databases."
            ),
            metadata={"filename": "test.txt", "page_number": 1},
        )
    ]


@pytest.fixture
def long_document() -> list[Document]:
    """Create a long document for testing chunk size limits."""
    paragraphs = [f"This is paragraph {i}. " * 20 for i in range(20)]
    return [
        Document(
            page_content="\n\n".join(paragraphs),
            metadata={"filename": "long.txt", "page_number": 1},
        )
    ]


class TestRecursiveChunking:
    """Tests for recursive character text splitting."""

    def test_basic_chunking(self, sample_documents: list[Document]) -> None:
        """Chunks should be created from sample documents."""
        chunks = chunk_documents(
            sample_documents,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=200,
            chunk_overlap=50,
        )
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunk_metadata(self, sample_documents: list[Document]) -> None:
        """Each chunk should have chunk_id and total_chunks metadata."""
        chunks = chunk_documents(
            sample_documents,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=200,
            chunk_overlap=50,
        )
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_chunk_size_limit(self, long_document: list[Document]) -> None:
        """Chunks should respect the maximum size."""
        chunks = chunk_documents(
            long_document,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=300,
            chunk_overlap=50,
        )
        for chunk in chunks:
            assert len(chunk.page_content) <= 400

    def test_small_document_single_chunk(self) -> None:
        """A very short document should produce at least one chunk."""
        docs = [Document(page_content="Hello world.", metadata={"filename": "short.txt"})]
        chunks = chunk_documents(docs, strategy=ChunkingStrategy.RECURSIVE, chunk_size=500)
        assert len(chunks) >= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_document(self) -> None:
        """Empty documents should produce no chunks."""
        docs = [Document(page_content="", metadata={"filename": "empty.txt"})]
        chunks = chunk_documents(docs, strategy=ChunkingStrategy.RECURSIVE, chunk_size=500)
        assert len(chunks) == 0

    def test_whitespace_only(self) -> None:
        """Whitespace-only content should produce no chunks."""
        docs = [Document(page_content="   \n\n   ", metadata={"filename": "ws.txt"})]
        chunks = chunk_documents(docs, strategy=ChunkingStrategy.RECURSIVE, chunk_size=500)
        assert len(chunks) == 0

    def test_multiple_documents(self) -> None:
        """Multiple documents should all be chunked."""
        docs = [
            Document(page_content="Doc one content. " * 20, metadata={"filename": "a.txt"}),
            Document(page_content="Doc two content. " * 20, metadata={"filename": "b.txt"}),
        ]
        chunks = chunk_documents(docs, strategy=ChunkingStrategy.RECURSIVE, chunk_size=200)
        filenames = {c.metadata.get("filename") for c in chunks}
        assert "a.txt" in filenames
        assert "b.txt" in filenames

    def test_empty_list(self) -> None:
        """Empty input should return empty output."""
        chunks = chunk_documents([])
        assert chunks == []
