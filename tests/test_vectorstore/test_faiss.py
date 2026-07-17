"""Tests for the FAISS vector store."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.config.settings import EmbeddingModelType, Settings
from app.embeddings.factory import get_embeddings_model
from app.vectorstore.faiss_store import FAISSVectorStore


@pytest.fixture
def embeddings():
    """Create a sentence-transformer embedding model for tests."""
    settings = Settings(embedding_model=EmbeddingModelType.BGE_SMALL, embedding_device="cpu")
    return get_embeddings_model(settings)


@pytest.fixture
def sample_docs() -> list[Document]:
    """Sample documents for vector store tests."""
    return [
        Document(
            page_content="Machine learning is a branch of artificial intelligence.",
            metadata={"filename": "ml.txt", "document_id": "doc-1", "page_number": 1},
        ),
        Document(
            page_content="Deep learning uses neural networks with many layers.",
            metadata={"filename": "dl.txt", "document_id": "doc-2", "page_number": 1},
        ),
        Document(
            page_content="Natural language processing deals with text and speech.",
            metadata={"filename": "nlp.txt", "document_id": "doc-3", "page_number": 1},
        ),
    ]


class TestFAISSVectorStore:
    """Tests for FAISS vector store operations."""

    @pytest.mark.slow
    def test_add_and_count(self, embeddings, sample_docs, tmp_path) -> None:
        """Adding documents should increase the count."""
        store = FAISSVectorStore(embedding_function=embeddings, persist_directory=str(tmp_path / "faiss"))
        ids = store.add_documents(sample_docs)
        assert len(ids) == 3
        assert store.get_document_count() == 3

    @pytest.mark.slow
    def test_similarity_search(self, embeddings, sample_docs, tmp_path) -> None:
        """Similarity search should return relevant results."""
        store = FAISSVectorStore(embedding_function=embeddings, persist_directory=str(tmp_path / "faiss"))
        store.add_documents(sample_docs)
        results = store.similarity_search("What is machine learning?", k=2)
        assert len(results) > 0
        assert len(results) <= 2
        assert all(isinstance(r, Document) for r in results)

    @pytest.mark.slow
    def test_similarity_search_with_score(self, embeddings, sample_docs, tmp_path) -> None:
        """Similarity search with score should return (Document, float) tuples."""
        store = FAISSVectorStore(embedding_function=embeddings, persist_directory=str(tmp_path / "faiss"))
        store.add_documents(sample_docs)
        results = store.similarity_search_with_score("What is machine learning?", k=2)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[0], Document) and isinstance(r[1], float) for r in results)

    @pytest.mark.slow
    def test_persistence(self, embeddings, sample_docs, tmp_path) -> None:
        """Index should be saveable and loadable."""
        persist_dir = str(tmp_path / "faiss_persist")
        store = FAISSVectorStore(embedding_function=embeddings, persist_directory=persist_dir)
        store.add_documents(sample_docs)
        store.save()
        store2 = FAISSVectorStore(embedding_function=embeddings, persist_directory=persist_dir)
        assert store2.get_document_count() == 3

    @pytest.mark.slow
    def test_delete_documents(self, embeddings, sample_docs, tmp_path) -> None:
        """Deleting documents should reduce the count."""
        store = FAISSVectorStore(embedding_function=embeddings, persist_directory=str(tmp_path / "faiss"))
        ids = store.add_documents(sample_docs)
        assert store.get_document_count() == 3
        store.delete_documents([ids[0]])
        assert store.get_document_count() == 2

    @pytest.mark.slow
    def test_empty_store(self, embeddings, tmp_path) -> None:
        """An empty store should return 0 count."""
        store = FAISSVectorStore(embedding_function=embeddings, persist_directory=str(tmp_path / "faiss_empty"))
        assert store.get_document_count() == 0

    @pytest.mark.slow
    def test_add_empty_list(self, embeddings, tmp_path) -> None:
        """Adding empty list should return empty list."""
        store = FAISSVectorStore(embedding_function=embeddings, persist_directory=str(tmp_path / "faiss"))
        ids = store.add_documents([])
        assert ids == []
