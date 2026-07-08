"""Qdrant vector store implementation.

Qdrant provides powerful filtering, payload indexing, and collection auto-create.
This implementation targets Qdrant Server (HTTP) for production reliability.

Usage:
    store = QdrantVectorStore(qdrant_url="http://localhost:6333")
    ids = store.add_documents(docs)
    results = store.similarity_search("query", k=5)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.documents import Document

from app.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-backed vector store with server persistence and filtering."""

    def __init__(
        self,
        embedding_function: Any = None,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "documents",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._embedding_function = embedding_function
        self._qdrant_url = qdrant_url
        self._collection_name = collection_name
        self._api_key = api_key
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Qdrant client via lazy import."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client_kwargs: dict[str, Any] = {"url": self._qdrant_url}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            self._client = QdrantClient(**client_kwargs)

            existing = [c.name for c in self._client.get_collections().collections]
            if self._collection_name not in existing:
                dim = getattr(self._embedding_function, "dimension", 384)
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                logger.info(
                    "qdrant_collection_created",
                    collection=self._collection_name,
                    dimension=dim,
                )
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install with: pip install qdrant-client"
            )

    def _qdrant_filter(self, filter_dict: dict[str, Any]) -> Any:
        """Convert a dict filter to Qdrant Filter object.

        Args:
            filter_dict: Simple {key: value} metadata filter.

        Returns:
            Qdrant Filter object.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = [
            FieldCondition(key=key, match=MatchValue(value=val))
            for key, val in filter_dict.items()
        ]
        return Filter(must=conditions)

    # ---- BaseVectorStore interface ----

    def add_documents(
        self,
        documents: list[Document],
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents with embeddings to the Qdrant collection.

        Args:
            documents: LangChain Document objects.
            collection_name: Optional collection override.

        Returns:
            List of assigned document IDs.
        """
        if not documents:
            return []

        from qdrant_client.models import PointStruct

        collection = collection_name or self._collection_name
        texts = [d.page_content for d in documents]
        vectors = self._embedding_function.embed_documents(texts)
        ids = [kwargs.get("ids", [None] * len(documents))[i] or f"doc_{i}" for i in range(len(documents))]

        points = [
            PointStruct(id=idx, vector=vec, payload={"text": text, **meta})
            for idx, (vec, text, meta) in enumerate(zip(vectors, texts, [d.metadata for d in documents]))
        ]
        self._client.upsert(collection_name=collection, points=points)
        logger.info("qdrant_documents_added", count=len(documents), collection=collection)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Find top-k most similar documents.

        Args:
            query: Search query text.
            k: Number of results.
            filter_dict: Optional metadata filter.

        Returns:
            List of Document objects.
        """
        results = self.similarity_search_with_score(query, k=k, filter_dict=filter_dict)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Find documents with relevance scores.

        Args:
            query: Search query text.
            k: Number of results.
            filter_dict: Optional metadata filter.

        Returns:
            List of (Document, score) tuples.
        """
        collection = collection_name or self._collection_name
        query_vector = self._embedding_function.embed_query(query)
        search_kwargs: dict[str, Any] = {"collection_name": collection, "query_vector": query_vector, "limit": k}

        if filter_dict:
            search_kwargs["query_filter"] = self._qdrant_filter(filter_dict)

        results = self._client.search(**search_kwargs)

        docs_with_scores: list[tuple[Document, float]] = []
        for hit in results:
            payload = hit.payload or {}
            doc = Document(
                page_content=payload.get("text", ""),
                metadata={pk: pv for pk, pv in payload.items() if pk != "text"},
            )
            docs_with_scores.append((doc, float(hit.score)))
        return docs_with_scores

    def delete_documents(
        self,
        ids: Optional[list[str]] = None,
        filter_dict: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """Delete documents by IDs or metadata filter.

        Args:
            ids: Optional document IDs to delete.
            filter_dict: Optional metadata filter.

        Returns:
            Number of documents deleted (0 if filter-based deletion).
        """
        collection = collection_name or self._collection_name
        if ids:
            int_ids: list[int] = []
            for i in ids:
                try:
                    clean = i.replace("doc_", "") if i.startswith("doc_") else i
                    int_ids.append(int(clean))
                except (ValueError, TypeError):
                    logger.warning("qdrant_invalid_id", doc_id=i)
                    continue
            if int_ids:
                self._client.delete(collection_name=collection, points_selector=int_ids)
            return len(ids)

        if filter_dict:
            self._client.delete(
                collection_name=collection,
                points_selector=self._qdrant_filter(filter_dict),
            )
            return 0

        return 0

    def get_document_count(self, collection_name: Optional[str] = None) -> int:
        """Return the number of stored documents."""
        collection = collection_name or self._collection_name
        info = self._client.get_collection(collection)
        return info.points_count or 0

    def reset(self, collection_name: Optional[str] = None) -> None:
        """Delete and recreate the collection."""
        collection = collection_name or self._collection_name
        self._client.delete_collection(collection)
        self._init_client()
        logger.info("qdrant_collection_reset", collection=collection)

    def get_collection_names(self) -> list[str]:
        """Return all available collection names."""
        return [c.name for c in self._client.get_collections().collections]

    def save(self, path: Optional[str] = None, **kwargs: Any) -> None:
        """No-op. Qdrant Server persists data automatically."""
        logger.debug("qdrant_save_noop", msg="Qdrant Server persists automatically.")

    def load(self, path: Optional[str] = None, **kwargs: Any) -> None:
        """No-op. Qdrant Server loads data automatically."""
        logger.debug("qdrant_load_noop", msg="Qdrant Server loads automatically.")
