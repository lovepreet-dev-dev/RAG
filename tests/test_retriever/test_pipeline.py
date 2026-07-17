"""Tests for the multi-stage retrieval pipeline."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from app.config.settings import Settings
from app.retriever.confidence import RetrievalResult
from app.retriever.rag_retriever import RetrievalPipeline
from app.vectorstore.faiss_store import FAISSVectorStore


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_embeddings():
    """Use fake embeddings for fast tests."""
    return FakeEmbeddings(size=128)


@pytest.fixture
def sample_docs() -> list[Document]:
    """Sample documents for retrieval tests."""
    return [
        Document(
            page_content="Python is a popular programming language for data science and machine learning.",
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
        Document(
            page_content="Deep learning uses neural networks with many layers.",
            metadata={"filename": "dl.txt", "document_id": "d4", "page_number": 1},
        ),
        Document(
            page_content="Natural language processing deals with text and speech.",
            metadata={"filename": "nlp.txt", "document_id": "d5", "page_number": 1},
        ),
    ]


@pytest.fixture
def populated_store(fake_embeddings, sample_docs, tmp_path) -> FAISSVectorStore:
    """Create a populated FAISS store for pipeline tests."""
    store = FAISSVectorStore(
        embedding_function=fake_embeddings,
        persist_directory=str(tmp_path / "faiss"),
    )
    store.add_documents(sample_docs)
    return store


@pytest.fixture
def pipeline(populated_store) -> RetrievalPipeline:
    """Create a RetrievalPipeline with default settings."""
    settings = Settings(
        enable_hybrid_search=True,
    )
    return RetrievalPipeline(
        vector_store=populated_store,
        llm=None,
        settings=settings,
    )


# ── Parallel retrieval ──────────────────────────────────────────────────────


class TestParallelRetrieval:
    """Tests for parallel multi-strategy retrieval."""

    def test_returns_results(self, pipeline) -> None:
        """Pipeline should return results."""
        results, confidence, _ = pipeline.retrieve("programming language")
        assert len(results) > 0
        assert confidence.score >= 0.0

    def test_multiple_strategies_used(self, pipeline) -> None:
        """Results should carry provenance from multiple strategies."""
        results, _, _ = pipeline.retrieve("programming language")
        strategies = {s for r in results for s in r.strategy.split(",")}
        assert len(strategies) > 1

    def test_results_are_retrieval_results(self, pipeline) -> None:
        """Results should be RetrievalResult objects."""
        results, _, _ = pipeline.retrieve("programming language")
        for r in results:
            assert isinstance(r, RetrievalResult)
            assert isinstance(r.document, Document)
            assert isinstance(r.score, float)
            assert isinstance(r.strategy, str)

    def test_deduplication(self, pipeline) -> None:
        """Duplicate documents should be removed."""
        results, _, _ = pipeline.retrieve("programming")
        contents = [r.document.page_content for r in results]
        assert len(contents) == len(set(contents))

    def test_score_threshold(self, pipeline) -> None:
        """Score threshold should filter low-scoring results."""
        results_high, _, _ = pipeline.retrieve("programming", score_threshold=0.99)
        results_low, _, _ = pipeline.retrieve("programming", score_threshold=0.0)
        assert len(results_high) <= len(results_low)


# ── Hybrid search ────────────────────────────────────────────────────────────


class TestHybridSearch:
    """Tests for hybrid search."""

    def test_hybrid_returns_results(self, pipeline) -> None:
        """Hybrid should return results."""
        results = pipeline._hybrid_search("programming", k=3, metadata_filter=None)
        assert len(results) > 0
        assert all(r.strategy == "hybrid" for r in results)


# ── Context selection ────────────────────────────────────────────────────────


class TestContextSelection:
    """Tests for final context selection with token budget."""

    def test_respects_token_budget(self, pipeline) -> None:
        """Selected results should fit within token budget."""
        results, _, _ = pipeline.retrieve("programming")
        selected = pipeline.select_final_context(results, max_tokens=100)
        # Verify all selected fit in budget
        from app.utils.helpers import estimate_tokens
        total = sum(estimate_tokens(r.document.page_content) for r in selected)
        assert total <= 100

    def test_prefers_diversity(self, pipeline) -> None:
        """Context selection should avoid duplicate content."""
        results, _, _ = pipeline.retrieve("programming")
        selected = pipeline.select_final_context(results)
        contents = [r.document.page_content for r in selected]
        assert len(contents) == len(set(contents))

    def test_empty_results(self, pipeline) -> None:
        """Empty results should return empty list."""
        selected = pipeline.select_final_context([])
        assert selected == []


# ── Score normalization ─────────────────────────────────────────────────────


class TestScoreNormalization:
    """Tests that scores are normalized across strategies."""

    def test_normalized_scores_in_0_1(self, pipeline) -> None:
        """All scores should be in [0, 1] after normalization."""
        results, _, _ = pipeline.retrieve("programming")
        for r in results:
            assert 0.0 <= r.score <= 1.0


# ── Confidence estimation ────────────────────────────────────────────────────


class TestConfidenceInPipeline:
    """Tests that confidence is computed correctly in the pipeline."""

    def test_confidence_in_response(self, pipeline) -> None:
        """Pipeline should return valid confidence."""
        _, confidence, _ = pipeline.retrieve("programming language")
        assert 0.0 <= confidence.score <= 1.0
        assert confidence.total_results > 0

    def test_confidence_with_strategies(self, pipeline) -> None:
        """Confidence should account for multi-strategy agreement."""
        _, confidence, _ = pipeline.retrieve("programming language")
        # With similarity + hybrid enabled, agreement should be > 0
        assert confidence.agreement_ratio > 0.0


# ── Legacy compatibility ─────────────────────────────────────────────────────


class TestLegacyCompatibility:
    """Tests for backward-compatible RAGRetriever interface."""

    def test_legacy_retrieve(self, populated_store) -> None:
        """Legacy retrieve should return (Document, score) tuples."""
        from app.retriever.rag_retriever import RAGRetriever

        settings = Settings()
        retriever = RAGRetriever(
            vector_store=populated_store,
            llm=None,
            settings=settings,
        )
        results = retriever.retrieve("programming language", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[0], Document) for r in results)
