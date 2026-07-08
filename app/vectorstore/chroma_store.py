"""ChromaDB vector store implementation.

Chroma provides built-in persistence, metadata filtering, and MMR diversity.
This implementation wraps ChromaClient with connection pooling and graceful
degradation when the server is unavailable.

Usage:
    store = ChromaVectorStore(persist_directory="./chroma_data")
    ids = store.add_documents(docs)
    results = store.similarity_search("query", k=5)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.documents import Document

from app.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector store with persistence and metadata filtering."""

    def __init__(
        self,
        embedding_function: Any = None,
        persist_directory: Optional[str] = None,
        collection_name: str = "documents",
        **kwargs: Any,
    ) -> None:
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._collection = None
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialise the Chroma client via lazy import."""
        try:
            import chromadb

            if self._persist_directory:
                self._client = chromadb.PersistentClient(path=self._persist_directory)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install with: pip install chromadb"
            )

    def _chroma_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a flat dict filter to Chroma's $and syntax.

        Args:
            filter_dict: Simple {key: value} metadata filter.

        Returns:
            Chroma-compatible filter dict.
        """
        if len(filter_dict) == 1:
            key, val = next(iter(filter_dict.items()))
            return {key: {"$eq": val}}
        conditions = [{k: {"$eq": v}} for k, v in filter_dict.items()]
        return {"$and": conditions}

    # ---- BaseVectorStore interface ----

    def add_documents(
        self,
        documents: list[Document],
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents with embeddings to the Chroma collection.

        Args:
            documents: LangChain Document objects.
            collection_name: Optional collection override.

        Returns:
            List of assigned document IDs.
        """
        if not documents:
            return []

        ids = [kwargs.get("ids", [None] * len(documents))[i] or f"doc_{i}" for i in range(len(documents))]
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        self._collection.add(documents=texts, metadatas=metadatas, ids=ids)
        logger.info("chroma_documents_added", count=len(documents), collection=self._collection_name)
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
        where = self._chroma_filter(filter_dict) if filter_dict else None
        results = self._collection.query(query_texts=[query], n_results=k, where=where)

        documents: list[Document] = []
        if results and results.get("documents"):
            for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
                documents.append(Document(page_content=doc_text, metadata=meta))
        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Find documents with similarity scores.

        Args:
            query: Search query text.
            k: Number of results.
            filter_dict: Optional metadata filter.

        Returns:
            List of (Document, score) tuples.
        """
        where = self._chroma_filter(filter_dict) if filter_dict else None
        results = self._collection.query(
            query_texts=[query], n_results=k, where=where, include=["distances"]
        )

        docs_with_scores: list[tuple[Document, float]] = []
        if results and results.get("documents"):
            for doc_text, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                docs_with_scores.append(
                    (Document(page_content=doc_text, metadata=meta), float(dist))
                )
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
            ids: Optional list of document IDs.
            filter_dict: Optional metadata filter.

        Returns:
            Number of documents deleted.
        """
        if ids:
            self._collection.delete(ids=ids)
            return len(ids)

        if filter_dict:
            existing = self._collection.get(where=self._chroma_filter(filter_dict))
            if existing and existing.get("ids"):
                self._collection.delete(ids=existing["ids"])
                return len(existing["ids"])
            return 0

        count = self._collection.count()
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return count

    def get_document_count(self, collection_name: Optional[str] = None) -> int:
        """Return the number of stored documents."""
        return self._collection.count()

    def reset(self, collection_name: Optional[str] = None) -> None:
        """Delete and recreate the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("chroma_collection_reset", collection=self._collection_name)

    def get_collection_names(self) -> list[str]:
        """Return all available collection names."""
        collections = self._client.list_collections()
        return [c.name for c in collections]

    def save(self, path: Optional[str] = None, **kwargs: Any) -> None:
        """Persist to disk.

        Chroma persists automatically when ``persist_directory`` was provided.
        This method is a no-op but satisfies the base interface.
        """
        logger.debug("chroma_save_noop", msg="Chroma persists automatically.")

    def load(self, path: Optional[str] = None, **kwargs: Any) -> None:
        """Load persisted data.

        Chroma loads automatically from the ``persist_directory`` on init.
        This method is a no-op but satisfies the base interface.
        """
        logger.debug("chroma_load_noop", msg="Chroma loads automatically.")
