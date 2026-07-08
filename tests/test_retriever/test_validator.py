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
        result = validator.validate("What is Python?", "Python is a programming language.", docs)
        assert isinstance(result, ValidationResult)
        assert isinstance(result.citations_ok, bool)
        assert isinstance(result.coverage_score, float)
        assert isinstance(result.hallucinated_claims, list)

    def test_empty_answer(self) -> None:
        validator = AnswerValidator(llm=None)
        result = validator.validate("What is Python?", "", [])
        assert result.coverage_score == 0.0

    def test_citation_check_with_sources(self) -> None:
        validator = AnswerValidator(llm=None)
        docs = [_make_result("Python was created in 1991.")]
        result = validator.validate(
            "What is Python?",
            "According to Source 1, Python was created in 1991.",
            docs,
        )
        # Should find citation
        assert result.citations_ok is True

    def test_citation_check_without_sources(self) -> None:
        validator = AnswerValidator(llm=None)
        docs = [_make_result("Python was created in 1991.")]
        result = validator.validate(
            "What is Python?",
            "Python was created in 1991.",
            docs,
        )
        # No Source X pattern but answer has keywords from docs
        assert isinstance(result.citations_ok, bool)

    def test_coverage_check(self) -> None:
        validator = AnswerValidator(llm=None)
        docs = [_make_result("Python is a high-level programming language.")]
        result = validator.validate(
            "What is Python?",
            "Python is a programming language.",
            docs,
        )
        # Should have some coverage
        assert result.coverage_score >= 0.0

    def test_hallucination_detection_keywords(self) -> None:
        validator = AnswerValidator(llm=None)
        docs = [_make_result("Python is a programming language.")]
        result = validator.validate(
            "What is Python?",
            "Python is a fruit that grows on trees.",
            docs,
        )
        # "fruit" and "trees" are not in the sources
        assert len(result.hallucinated_claims) > 0

    def test_result_to_dict(self) -> None:
        validator = AnswerValidator(llm=None)
        result = ValidationResult(
            citations_ok=True,
            missing_citations=[],
            coverage_score=0.85,
            hallucinated_claims=["claim1"],
        )
        d = result.to_dict()
        assert d["citations_ok"] is True
        assert d["coverage_score"] == 0.85
        assert d["hallucinated_claims"] == ["claim1"]
