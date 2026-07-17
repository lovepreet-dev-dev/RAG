"""Factory for creating vector store instances.

FAISS is the single supported backend: it is local, fast, and needs no
external service.  The factory indirection is kept so an additional backend
can be reintroduced without touching call sites.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.config.settings import Settings, VectorStoreType, get_settings
from app.utils.logging import get_logger
from app.vectorstore.base import BaseVectorStore

logger = get_logger(__name__)


def get_vector_store(
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> BaseVectorStore:
    """Create the configured vector store backend.

    Args:
        embeddings: Embedding model instance.  If ``None``, a new one is created.
        settings: Optional settings override.

    Returns:
        Configured ``BaseVectorStore`` implementation.

    Raises:
        ValueError: If the vector store type is not supported.
    """
    if settings is None:
        settings = get_settings()

    if embeddings is None:
        from app.embeddings.factory import get_embeddings_model

        embeddings = get_embeddings_model(settings)

    store_type = settings.vector_store
    logger.info("creating_vector_store", store_type=store_type.value)

    if store_type == VectorStoreType.FAISS:
        from app.vectorstore.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(
            embedding_function=embeddings,
            persist_directory=settings.faiss_index_dir,
        )

        # Check embedding version compatibility
        _check_embedding_version(settings, settings.faiss_index_dir)

        return store

    raise ValueError(f"Unsupported vector store type: {store_type}")


def _check_embedding_version(settings: Settings, persist_dir: str) -> None:
    """Log a warning if the embedding model has changed since last index."""
    from app.embeddings.versioning import EmbeddingVersionTracker

    tracker = EmbeddingVersionTracker(persist_dir)
    needs_reindex, reason = tracker.needs_reindex(
        current_model=settings.embedding_model.value,
    )
    if needs_reindex:
        logger.warning(
            "embedding_model_changed",
            reason=reason,
            action="Consider reindexing documents",
        )
