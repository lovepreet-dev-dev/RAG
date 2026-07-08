"""Tests for QueryPreprocessor."""

import pytest

from app.retriever.preprocessing import QueryPreprocessor


@pytest.fixture
def preprocessor() -> QueryPreprocessor:
    return QueryPreprocessor()


class TestQueryPreprocessor:
    """Tests for the query preprocessor."""

    def test_basic_preprocessing(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.preprocess("What is machine learning?")
        assert result.original == "What is machine learning?"
        assert len(result.normalized) > 0
        assert result.normalized == result.normalized.strip()

    def test_normalizes_whitespace(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.preprocess("  What   is   machine   learning?  ")
        assert "  " not in result.normalized

    def test_normalizes_punctuation(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.preprocess("What is machine learning???")
        assert result.normalized.endswith("?")
        assert "???" not in result.normalized

    def test_extracts_metadata_filters(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.preprocess("Find document.pdf for question about AI")
        assert "filename" in result.metadata_filters

    def test_extracts_page_number_filter(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.preprocess("What is on page 5?")
        assert "page_number" in result.metadata_filters
        assert result.metadata_filters["page_number"] == 5

    def test_empty_query(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.preprocess("")
        assert result.normalized == ""
        assert result.metadata_filters == {}

    def test_context_enrichment(self, preprocessor: QueryPreprocessor) -> None:
        chat_history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        result = preprocessor.preprocess(
            "Tell me more about it", chat_history=chat_history,
        )
        assert len(result.normalized) > len("tell me more about it")

    def test_language_detection(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.preprocess("What is RAG?")
        assert result.language == "en"

    def test_strip_trailing_punctuation(self, preprocessor: QueryPreprocessor) -> None:
        result = preprocessor.preprocess("What is this.")
        assert result.normalized.endswith("this") or not result.normalized.endswith(".")
