"""Tests for the RAG retriever and pipeline."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from app.config.settings import Settings
from app.retriever.rag_retriever import RAGRetriever, RetrievalPipeline
from app.vectorstore.faiss_store import FAISSVectorStore


@pytest.fixture
def fake_embeddings():
    """Use fake embeddings for fast tests."""
    return FakeEmbeddings(size=128)


@pytest.fixture
def populated_store(fake_embeddings, tmp_path) -> FAISSVectorStore:
    """Create a populated FAISS store for retrieval tests."""
    store = FAISSVectorStore(
        embedding_function=fake_embeddings,
        persist_directory=str(tmp_path / "faiss"),
    )
    docs = [
        Document(
            page_content="Python is a popular programming language for data science.",
            metadata={"filename": "python.txt", "document_id": "d1", "page_number": 1},
        ),
        Document(
            page_content="JavaScript is used for web development and browser scripting.",
            metadata={"filename": "js.txt", "document_id": "d2", "page_number": 1},
        ),
        Document(
            page_content="Machine learning models require large datasets for training.",
            metadata={"filename": "ml.txt", "document_id": "d3", "page_number": 1},
        ),
    ]
    store.add_documents(docs)
    return store


class TestRAGRetriever:
    """Tests for the legacy RAG retriever interface."""

    def test_similarity_retrieval(self, populated_store) -> None:
        """Similarity search should return results."""
        settings = Settings()
        retriever = RAGRetriever(vector_store=populated_store, settings=settings)
        results = retriever.retrieve(
            "programming language", top_k=2, retrieval_mode="similarity",
        )
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)

    def test_retrieval_respects_top_k(self, populated_store) -> None:
        """Results should not exceed top_k."""
        settings = Settings()
        retriever = RAGRetriever(vector_store=populated_store, settings=settings)
        results = retriever.retrieve("data", top_k=1, retrieval_mode="similarity")
        assert len(results) <= 1

    def test_metadata_filter(self, populated_store) -> None:
        """Metadata filter should narrow results."""
        settings = Settings()
        retriever = RAGRetriever(vector_store=populated_store, settings=settings)
        results = retriever.retrieve(
            "programming",
            top_k=5,
            retrieval_mode="similarity",
            metadata_filter={"filename": "python.txt"},
        )
        for doc, _ in results:
            assert doc.metadata.get("filename") == "python.txt"

    def test_score_threshold(self, populated_store) -> None:
        """Score threshold should filter out low-scoring results."""
        settings = Settings()
        retriever = RAGRetriever(vector_store=populated_store, settings=settings)
        results = retriever.retrieve(
            "programming language",
            top_k=5,
            score_threshold=999.0,
            retrieval_mode="similarity",
        )
        assert len(results) == 0


class TestRetrievalPipeline:
    """Tests for the multi-stage retrieval pipeline."""

    def test_pipeline_returns_results(self, populated_store) -> None:
        """Pipeline should return results and confidence."""
        settings = Settings()
        pipeline = RetrievalPipeline(
            vector_store=populated_store,
            llm=None,
            settings=settings,
        )
        results, confidence, _ = pipeline.retrieve("programming language")
        assert len(results) > 0
        assert 0.0 <= confidence.score <= 1.0

    def test_pipeline_deduplicates(self, populated_store) -> None:
        """Pipeline should deduplicate results."""
        settings = Settings()
        pipeline = RetrievalPipeline(
            vector_store=populated_store,
            llm=None,
            settings=settings,
        )
        results, _, _ = pipeline.retrieve("programming")
        contents = [r.document.page_content for r in results]
        assert len(contents) == len(set(contents))

    def test_pipeline_respects_token_budget(self, populated_store) -> None:
        """Pipeline context selection should respect token budget."""
        settings = Settings()
        pipeline = RetrievalPipeline(
            vector_store=populated_store,
            llm=None,
            settings=settings,
        )
        results, _, _ = pipeline.retrieve("programming")
        selected = pipeline.select_final_context(results, max_tokens=50)
        from app.utils.helpers import estimate_tokens
        total = sum(estimate_tokens(r.document.page_content) for r in selected)
        assert total <= 50
