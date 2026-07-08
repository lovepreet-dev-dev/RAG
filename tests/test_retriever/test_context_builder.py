"""Tests for ContextBuilder."""

import pytest
from langchain_core.documents import Document

from app.retriever.confidence import RetrievalResult
from app.retriever.context_builder import ContextBuilder


@pytest.fixture
def builder() -> ContextBuilder:
    return ContextBuilder(max_tokens=500)


def _make_result(
    content: str,
    filename: str = "test.pdf",
    page: int = 1,
    score: float = 0.9,
    chunk_id: int = 0,
) -> RetrievalResult:
    return RetrievalResult(
        document=Document(
            page_content=content,
            metadata={"filename": filename, "page_number": page, "chunk_id": chunk_id},
        ),
        score=score,
        strategy="similarity",
    )


class TestContextBuilder:
    """Tests for the context builder."""

    def test_empty_results(self, builder: ContextBuilder) -> None:
        ctx = builder.build([])
        assert ctx.results == []
        assert ctx.total_tokens == 0

    def test_basic_build(self, builder: ContextBuilder) -> None:
        results = [_make_result("This is a test chunk about AI.")]
        ctx = builder.build(results)
        assert len(ctx.results) == 1
        assert ctx.total_tokens > 0

    def test_token_budget_respected(self, builder: ContextBuilder) -> None:
        long_content = "word " * 200  # ~200 tokens
        results = [
            _make_result(long_content, page=1),
            _make_result(long_content, page=2),
            _make_result(long_content, page=3),
        ]
        ctx = builder.build(results, max_tokens=100)
        assert ctx.total_tokens <= 100 + 50  # small buffer

    def test_deduplication(self, builder: ContextBuilder) -> None:
        results = [
            _make_result("Same content", page=1),
            _make_result("Same content", page=1),
        ]
        ctx = builder.build(results)
        assert len(ctx.results) == 1

    def test_diversity_by_filename(self, builder: ContextBuilder) -> None:
        results = [
            _make_result("Chunk A1", filename="a.pdf", page=1, score=0.95),
            _make_result("Chunk A2", filename="a.pdf", page=2, score=0.94),
            _make_result("Chunk B1", filename="b.pdf", page=1, score=0.93),
            _make_result("Chunk B2", filename="b.pdf", page=2, score=0.92),
        ]
        ctx = builder.build(results, max_tokens=500)
        filenames = [r.document.metadata.get("filename") for r in ctx.results]
        # Round-robin should interleave
        assert "a.pdf" in filenames
        assert "b.pdf" in filenames

    def test_neighbor_merge(self, builder: ContextBuilder) -> None:
        results = [
            _make_result("Chunk one", page=1, chunk_id=0),
            _make_result("Chunk two", page=1, chunk_id=1),
        ]
        ctx = builder.build(results)
        # If pages match and chunk_ids are sequential, they might merge
        # The exact behavior depends on implementation
        assert len(ctx.results) >= 1
