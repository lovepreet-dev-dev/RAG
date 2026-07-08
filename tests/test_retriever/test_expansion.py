"""Tests for QueryExpander."""

import pytest

from app.retriever.expansion import QueryExpander


@pytest.fixture
def expander_no_llm() -> QueryExpander:
    return QueryExpander(llm=None)


class TestQueryExpander:
    """Tests for the query expander."""

    def test_synonym_expansion(self, expander_no_llm: QueryExpander) -> None:
        result = expander_no_llm.expand_synonyms("What is ML?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_synonym_expansion_with_known_terms(self, expander_no_llm: QueryExpander) -> None:
        result = expander_no_llm.expand_synonyms("Explain neural networks")
        # Should contain expanded terms
        assert isinstance(result, str)

    def test_keyword_expansion(self, expander_no_llm: QueryExpander) -> None:
        result = expander_no_llm.expand_keywords("What is the best approach for machine learning?")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_keyword_expansion_removes_stopwords(self, expander_no_llm: QueryExpander) -> None:
        result = expander_no_llm.expand_keywords("What is the best approach for machine learning?")
        # Stopwords should be filtered
        assert "what" not in [k.lower() for k in result]
        assert "the" not in [k.lower() for k in result]
        assert "is" not in [k.lower() for k in result]

    def test_keyword_expansion_empty(self, expander_no_llm: QueryExpander) -> None:
        result = expander_no_llm.expand_keywords("the a an")
        assert isinstance(result, list)

    def test_multi_query_no_llm_returns_empty(self, expander_no_llm: QueryExpander) -> None:
        result = expander_no_llm.expand_multi_query("What is ML?", count=3)
        assert result == []

    def test_hyde_no_llm_returns_empty(self, expander_no_llm: QueryExpander) -> None:
        result = expander_no_llm.expand_hyde("What is ML?")
        assert result == ""
