"""Tests for QueryClassifier."""

import pytest

from app.retriever.classifier import ClassifiedQuery, QueryClassifier, QueryType


@pytest.fixture
def classifier() -> QueryClassifier:
    return QueryClassifier()


class TestQueryClassifier:
    """Tests for the query classifier."""

    def test_factual_query(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("What is the capital of France?")
        assert result.query_type in (QueryType.FACTUAL, QueryType.UNKNOWN)
        assert 0.0 <= result.confidence <= 1.0

    def test_conversational_query(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("Hello, how are you?")
        assert result.query_type in (QueryType.CONVERSATIONAL, QueryType.UNKNOWN)

    def test_summarization_query(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("Summarize this document")
        assert result.query_type in (QueryType.SUMMARIZATION, QueryType.UNKNOWN)

    def test_comparison_query(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("Compare Python and Java")
        assert result.query_type in (QueryType.COMPARISON, QueryType.UNKNOWN)

    def test_reasoning_query(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("Why does the sky appear blue?")
        assert result.query_type in (QueryType.REASONING, QueryType.UNKNOWN)

    def test_code_query(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("Write a Python function to sort a list")
        assert result.query_type in (QueryType.CODE, QueryType.UNKNOWN)

    def test_math_query(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("Calculate the derivative of x^2")
        assert result.query_type in (QueryType.MATH, QueryType.UNKNOWN)

    def test_has_suggested_strategies(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("What is machine learning?")
        assert isinstance(result.suggested_strategies, list)
        assert len(result.suggested_strategies) > 0

    def test_has_suggested_top_k(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("What is machine learning?")
        assert isinstance(result.suggested_top_k, int)
        assert result.suggested_top_k > 0

    def test_empty_query(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("")
        assert result.query_type == QueryType.CONVERSATIONAL
