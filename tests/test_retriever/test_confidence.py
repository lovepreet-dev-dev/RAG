"""Tests for the confidence estimation module."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.retriever.confidence import (
    RetrievalConfidence,
    RetrievalResult,
    compute_answer_confidence,
    compute_retrieval_confidence,
    deduplicate_results,
    normalize_scores,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_result(
    content: str = "test content",
    score: float = 0.8,
    strategy: str = "similarity",
    filename: str = "test.txt",
) -> RetrievalResult:
    """Helper to create a RetrievalResult."""
    return RetrievalResult(
        document=Document(
            page_content=content,
            metadata={"filename": filename, "page_number": 1},
        ),
        score=score,
        strategy=strategy,
    )


# ── compute_retrieval_confidence ─────────────────────────────────────────────


class TestRetrievalConfidence:
    """Tests for retrieval confidence computation."""

    def test_empty_results(self) -> None:
        """Empty results should produce zero confidence."""
        conf = compute_retrieval_confidence([])
        assert conf.score == 0.0
        assert conf.total_results == 0
        assert conf.unique_documents == 0

    def test_single_strategy_results(self) -> None:
        """Results from a single strategy should have low agreement."""
        results = [
            _make_result("doc1", 0.9, "similarity"),
            _make_result("doc2", 0.8, "similarity"),
            _make_result("doc3", 0.7, "similarity"),
        ]
        conf = compute_retrieval_confidence(results)
        assert conf.total_results == 3
        assert conf.agreement_ratio == pytest.approx(1 / 2, abs=0.01)
        assert conf.score > 0.0

    def test_multi_strategy_results(self) -> None:
        """Results from multiple strategies should have higher agreement."""
        results = [
            _make_result("doc1", 0.9, "similarity"),
            _make_result("doc2", 0.8, "hybrid"),
            _make_result("doc3", 0.7, "similarity,hybrid"),
        ]
        conf = compute_retrieval_confidence(results)
        assert conf.agreement_ratio == pytest.approx(1.0, abs=0.01)
        assert conf.strategy_counts == {"similarity": 2, "hybrid": 2}

    def test_high_scores_increase_confidence(self) -> None:
        """Higher scores should produce higher confidence."""
        high_results = [_make_result(f"doc{i}", 0.95, "similarity") for i in range(5)]
        low_results = [_make_result(f"doc{i}", 0.2, "similarity") for i in range(5)]

        high_conf = compute_retrieval_confidence(high_results)
        low_conf = compute_retrieval_confidence(low_results)
        assert high_conf.score > low_conf.score

    def test_diverse_sources_increase_confidence(self) -> None:
        """More diverse filenames should increase metadata consistency."""
        diverse = [
            _make_result("doc1", 0.8, "similarity", filename=f"file{i}.txt")
            for i in range(5)
        ]
        same_source = [
            _make_result("doc1", 0.8, "similarity", filename="same.txt")
            for _ in range(5)
        ]
        diverse_conf = compute_retrieval_confidence(diverse)
        same_conf = compute_retrieval_confidence(same_source)
        assert diverse_conf.metadata_consistency > same_conf.metadata_consistency

    def test_score_bounds(self) -> None:
        """Confidence score should be between 0 and 1."""
        results = [_make_result("doc", 0.5, "similarity")]
        conf = compute_retrieval_confidence(results)
        assert 0.0 <= conf.score <= 1.0


# ── compute_answer_confidence ────────────────────────────────────────────────


class TestAnswerConfidence:
    """Tests for answer confidence computation."""

    def test_low_confidence(self) -> None:
        """Low scores should produce 'low' level."""
        retrieval = RetrievalConfidence(
            score=0.1, avg_score=0.1, score_variance=0.01,
            agreement_ratio=0.33, document_diversity=0.5,
            metadata_consistency=0.5, chunk_overlap_ratio=0.0,
            total_results=3,
            unique_documents=2, strategy_counts={"similarity": 3},
        )
        conf = compute_answer_confidence(retrieval)
        assert conf.level == "low"
        assert conf.score < 0.4

    def test_high_confidence(self) -> None:
        """High scores should produce 'high' level."""
        retrieval = RetrievalConfidence(
            score=0.95, avg_score=0.95, score_variance=0.001,
            agreement_ratio=1.0, document_diversity=1.0,
            metadata_consistency=1.0, chunk_overlap_ratio=0.8,
            total_results=9,
            unique_documents=9, strategy_counts={"similarity": 3, "mmr": 3, "hybrid": 3},
        )
        conf = compute_answer_confidence(
            retrieval,
            reranker_scores=[0.95, 0.92],
            doc_count=5,
            source_agreement=1.0,
            citation_coverage=0.8,
        )
        assert conf.level == "high"
        assert conf.score >= 0.7

    def test_medium_confidence(self) -> None:
        """Medium scores should produce 'medium' level."""
        retrieval = RetrievalConfidence(
            score=0.7, avg_score=0.7, score_variance=0.03,
            agreement_ratio=0.66, document_diversity=0.7,
            metadata_consistency=0.7, chunk_overlap_ratio=0.5,
            total_results=6,
            unique_documents=4, strategy_counts={"similarity": 3, "mmr": 3},
        )
        conf = compute_answer_confidence(
            retrieval, doc_count=3, reranker_scores=[0.9],
        )
        assert conf.level == "medium"

    def test_reranker_boosts_confidence(self) -> None:
        """Reranker scores should increase confidence."""
        retrieval = RetrievalConfidence(
            score=0.5, avg_score=0.5, score_variance=0.05,
            agreement_ratio=0.66, document_diversity=0.7,
            metadata_consistency=0.7, chunk_overlap_ratio=0.5,
            total_results=6,
            unique_documents=4, strategy_counts={"similarity": 3, "mmr": 3},
        )
        no_rerank = compute_answer_confidence(retrieval)
        with_rerank = compute_answer_confidence(retrieval, reranker_scores=[0.9, 0.85])
        assert with_rerank.score >= no_rerank.score

    def test_score_bounds(self) -> None:
        """Confidence score should be between 0 and 1."""
        retrieval = RetrievalConfidence(
            score=0.5, avg_score=0.5, score_variance=0.05,
            agreement_ratio=0.5, document_diversity=0.5,
            metadata_consistency=0.5, chunk_overlap_ratio=0.5,
            total_results=5,
            unique_documents=3, strategy_counts={"similarity": 5},
        )
        conf = compute_answer_confidence(retrieval)
        assert 0.0 <= conf.score <= 1.0


# ── deduplicate_results ──────────────────────────────────────────────────────


class TestDeduplication:
    """Tests for result deduplication."""

    def test_no_duplicates(self) -> None:
        """Unique documents should all remain."""
        results = [
            _make_result("doc1", 0.9),
            _make_result("doc2", 0.8),
            _make_result("doc3", 0.7),
        ]
        deduped = deduplicate_results(results)
        assert len(deduped) == 3

    def test_removes_duplicates(self) -> None:
        """Duplicate content should be merged, keeping highest score."""
        results = [
            _make_result("same content", 0.9, "similarity"),
            _make_result("same content", 0.7, "mmr"),
            _make_result("different content", 0.8, "hybrid"),
        ]
        deduped = deduplicate_results(results)
        assert len(deduped) == 2
        assert deduped[0].score == 0.9  # Highest score kept

    def test_sorted_by_score(self) -> None:
        """Results should be sorted by score descending."""
        results = [
            _make_result("low", 0.3),
            _make_result("high", 0.9),
            _make_result("mid", 0.6),
        ]
        deduped = deduplicate_results(results)
        scores = [r.score for r in deduped]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self) -> None:
        """Empty input should return empty list."""
        assert deduplicate_results([]) == []


# ── normalize_scores ─────────────────────────────────────────────────────────


class TestScoreNormalization:
    """Tests for score normalization."""

    def test_normalizes_to_0_1(self) -> None:
        """Scores should be normalized to [0, 1] range."""
        results = [
            _make_result("doc1", 10.0),
            _make_result("doc2", 20.0),
            _make_result("doc3", 30.0),
        ]
        normalized = normalize_scores(results)
        scores = [r.score for r in normalized]
        assert min(scores) == pytest.approx(0.0)
        assert max(scores) == pytest.approx(1.0)

    def test_same_scores(self) -> None:
        """All same scores should normalize to 1.0."""
        results = [
            _make_result("doc1", 5.0),
            _make_result("doc2", 5.0),
        ]
        normalized = normalize_scores(results)
        assert all(r.score == pytest.approx(1.0) for r in normalized)

    def test_empty_input(self) -> None:
        """Empty input should return empty list."""
        assert normalize_scores([]) == []

    def test_preserves_strategy(self) -> None:
        """Strategy should be preserved through normalization."""
        results = [_make_result("doc1", 10.0, strategy="mmr")]
        normalized = normalize_scores(results)
        assert normalized[0].strategy == "mmr"
