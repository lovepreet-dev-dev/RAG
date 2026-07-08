"""Factory for creating vector store instances.

Switching the backend requires only changing the ``VECTOR_STORE``
environment variable — no code changes needed.
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

        return FAISSVectorStore(
            embedding_function=embeddings,
            persist_directory=settings.faiss_index_dir,
        )

    if store_type == VectorStoreType.CHROMA:
        from app.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection,
        )

    if store_type == VectorStoreType.QDRANT:
        from app.vectorstore.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(
            embedding_function=embeddings,
            qdrant_url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
            collection_name=settings.qdrant_collection,
        )

    if store_type == VectorStoreType.PINECONE:
        from app.vectorstore.pinecone_store import PineconeVectorStore

        return PineconeVectorStore(
            embedding_function=embeddings,
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index,
            environment=settings.pinecone_environment,
        )

    raise ValueError(f"Unsupported vector store type: {store_type}")
