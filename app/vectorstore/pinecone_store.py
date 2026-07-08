"""Pinecone vector store implementation.

Pinecone provides a fully-managed cloud vector database with automatic
index creation, namespace support, and built-in metadata filtering.

Usage:
    store = PineconeVectorStore(api_key="...", index_name="my-index")
    ids = store.add_documents(docs)
    results = store.similarity_search("query", k=5)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.documents import Document

from app.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class PineconeVectorStore(BaseVectorStore):
    """Pinecone-backed vector store with managed cloud persistence."""

    def __init__(
        self,
        embedding_function: Any = None,
        api_key: Optional[str] = None,
        index_name: str = "rag-documents",
        namespace: str = "default",
        environment: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._embedding_function = embedding_function
        self._api_key = api_key
        self._index_name = index_name
        self._namespace = namespace
        self._environment = environment
        self._index = None
        self._pc = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Pinecone client via lazy import."""
        try:
            from pinecone import Pinecone, ServerlessSpec

            pc_kwargs: dict[str, Any] = {}
            if self._api_key:
                pc_kwargs["api_key"] = self._api_key

            self._pc = Pinecone(**pc_kwargs)
            existing = [idx.name for idx in self._pc.list_indexes()]

            if self._index_name not in existing:
                spec = ServerlessSpec(cloud="aws", region=self._environment or "us-east-1")
                self._pc.create_index(
                    name=self._index_name,
                    dimension=getattr(self._embedding_function, "dimension", 384),
                    metric="cosine",
                    spec=spec,
                )
                logger.info(
                    "pinecone_index_created",
                    index=self._index_name,
                    environment=self._environment,
                )

            self._index = self._pc.Index(self._index_name)
        except ImportError:
            raise ImportError(
                "pinecone-client is required for PineconeVectorStore. "
                "Install with: pip install pinecone-client"
            )

    # ---- BaseVectorStore interface ----

    def add_documents(
        self,
        documents: list[Document],
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents with embeddings to the Pinecone index.

        Args:
            documents: LangChain Document objects.
            collection_name: Treated as the namespace override.

        Returns:
            List of assigned document IDs.
        """
        if not documents:
            return []

        namespace = collection_name or self._namespace
        texts = [d.page_content for d in documents]
        vectors = self._embedding_function.embed_documents(texts)
        ids = [kwargs.get("ids", [None] * len(documents))[i] or f"doc_{i}" for i in range(len(documents))]

        upsert_data = [
            {
                "id": doc_id,
                "values": vec,
                "metadata": {"text": text, **meta},
            }
            for doc_id, vec, text, meta in zip(
                ids, vectors, texts, [d.metadata for d in documents]
            )
        ]
        self._index.upsert(vectors=upsert_data, namespace=namespace)
        logger.info("pinecone_documents_added", count=len(documents), namespace=namespace)
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
        results = self.similarity_search_with_score(
            query, k=k, filter_dict=filter_dict, collection_name=collection_name
        )
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
        namespace = collection_name or self._namespace
        query_vector = self._embedding_function.embed_query(query)

        search_kwargs: dict[str, Any] = {
            "vector": query_vector,
            "top_k": k,
            "namespace": namespace,
            "include_metadata": True,
        }
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        response = self._index.query(**search_kwargs)

        docs_with_scores: list[tuple[Document, float]] = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            doc = Document(
                page_content=metadata.pop("text", ""),
                metadata=metadata,
            )
            docs_with_scores.append((doc, float(match.get("score", 0.0))))
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
            Number of documents deleted (0 for filter-based deletion).
        """
        namespace = collection_name or self._namespace
        if ids:
            self._index.delete(ids=ids, namespace=namespace)
            return len(ids)

        if filter_dict:
            self._index.delete(filter=filter_dict, namespace=namespace)
            return 0

        return 0

    def get_document_count(self, collection_name: Optional[str] = None) -> int:
        """Return the number of stored documents (approximate)."""
        namespace = collection_name or self._namespace
        stats = self._index.describe_namespace_stats(namespace=namespace)
        return stats.get("vector_count", 0)

    def reset(self, collection_name: Optional[str] = None) -> None:
        """Delete all vectors in the namespace."""
        namespace = collection_name or self._namespace
        self._index.delete(delete_all=True, namespace=namespace)
        logger.info("pinecone_namespace_reset", namespace=namespace)

    def get_collection_names(self) -> list[str]:
        """Return the default namespace as the collection name."""
        return [self._namespace]

    def save(self, path: Optional[str] = None, **kwargs: Any) -> None:
        """No-op. Pinecone persists data automatically in the cloud."""
        logger.debug("pinecone_save_noop", msg="Pinecone persists automatically.")

    def load(self, path: Optional[str] = None, **kwargs: Any) -> None:
        """No-op. Pinecone loads data automatically from the cloud."""
        logger.debug("pinecone_load_noop", msg="Pinecone loads automatically.")
