"""Tests for AnswerValidator."""

import pytest
from langchain_core.documents import Document

from app.retriever.confidence import RetrievalResult
from app.retriever.validator import AnswerValidator, ValidationResult


def _make_result(content: str, filename: str = "test.pdf") -> RetrievalResult:
    return RetrievalResult(
        document=Document(
            page_content=content,
            metadata={"filename": filename, "page_number": 1},
        ),
        score=0.9,
        strategy="similarity",
    )


class TestAnswerValidator:
    """Tests for the answer validator."""

    def test_no_llm_keyword_fallback(self) -> None:
        validator = AnswerValidator(llm=None)
        docs = [_make_result("Python is a programming language created by Guido van Rossum.")]
        result = validator.validate(
            answer="Python is a programming language.",
            sources=docs,
            query="What is Python?",
        )
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.coverage_score, float)
        assert isinstance(result.unsupported_statements, list)

    def test_empty_answer(self) -> None:
        validator = AnswerValidator(llm=None)
        result = validator.validate(answer="", sources=[], query="What is Python?")
        assert result.coverage_score == 0.0
        assert result.is_valid is False

    def test_citation_check_with_sources(self) -> None:
        validator = AnswerValidator(llm=None)
        docs = [_make_result("Python was created in 1991.")]
        result = validator.validate(
            answer="According to [Source 1: test.pdf, Page 1], Python was created in 1991.",
            sources=docs,
            query="What is Python?",
        )
        # Should have valid citations
        assert result.citation_score > 0.0

    def test_citation_check_no_citations(self) -> None:
        validator = AnswerValidator(llm=None)
        docs = [_make_result("Python was created in 1991.")]
        result = validator.validate(
            answer="Python was created in 1991.",
            sources=docs,
            query="What is Python?",
        )
        # No explicit citations found
        assert result.citation_score == 0.7  # mild penalty

    def test_coverage_check(self) -> None:
        validator = AnswerValidator(llm=None)
        docs = [_make_result("Python is a high-level programming language.")]
        result = validator.validate(
            answer="Python is a programming language.",
            sources=docs,
            query="What is Python?",
        )
        assert result.coverage_score >= 0.0

    def test_hallucination_detection_keywords(self) -> None:
        validator = AnswerValidator(llm=None)
        docs = [_make_result("Python is a programming language.")]
        result = validator.validate(
            answer="Python is a fruit that grows on trees.",
            sources=docs,
            query="What is Python?",
        )
        # "fruit" and "trees" are not in the sources
        assert len(result.unsupported_statements) > 0

    def test_result_to_dict(self) -> None:
        result = ValidationResult(
            is_valid=True,
            citation_score=0.9,
            coverage_score=0.85,
            hallucination_score=0.95,
            overall_score=0.9,
            issues=[],
            unsupported_statements=["claim1"],
        )
        d = result.to_dict()
        assert d["is_valid"] is True
        assert d["coverage_score"] == 0.85
        assert d["unsupported_statements"] == ["claim1"]
        assert "citation_score" in d
        assert "hallucination_score" in d
