"""Tests for embedding model factory."""

from __future__ import annotations

import pytest

from app.config.settings import EmbeddingModelType, Settings
from app.embeddings.factory import get_embeddings_model


class TestEmbeddingFactory:
    """Tests for the embedding factory."""

    @pytest.mark.slow
    def test_bge_small_creation(self) -> None:
        """BGE-small model should be created successfully."""
        settings = Settings(embedding_model=EmbeddingModelType.BGE_SMALL, embedding_device="cpu")
        model = get_embeddings_model(settings)
        assert model is not None
        result = model.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.slow
    def test_batch_embedding(self) -> None:
        """Batch embedding should return correct number of vectors."""
        settings = Settings(embedding_model=EmbeddingModelType.BGE_SMALL, embedding_device="cpu")
        model = get_embeddings_model(settings)
        texts = ["Hello world", "Machine learning", "Deep learning"]
        results = model.embed_documents(texts)
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    @pytest.mark.slow
    def test_embedding_consistency(self) -> None:
        """Same input should produce same embedding."""
        settings = Settings(embedding_model=EmbeddingModelType.BGE_SMALL, embedding_device="cpu")
        model = get_embeddings_model(settings)
        text = "Consistent embedding test"
        result1 = model.embed_query(text)
        result2 = model.embed_query(text)
        assert result1 == result2
