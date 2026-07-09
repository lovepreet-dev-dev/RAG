"""Abstract vector store interface.

All vector store implementations must inherit from BaseVectorStore
and implement every abstract method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class BaseVectorStore(VectorStore, ABC):
    """Abstract base for all vector store implementations.

    Subclasses must implement the abstract methods defined here.
    Optional methods provide default fallback behaviour (e.g. hybrid_search
    falls back to similarity_search when not natively supported).
    """

    @abstractmethod
    def add_documents(
        self,
        documents: list[Document],
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents with embeddings to the vector store.

        Args:
            documents: LangChain Document objects with text and metadata.
            collection_name: Optional override for the target collection.

        Returns:
            List of document IDs assigned by the store.
        """

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Find the top-k most similar documents.

        Args:
            query: Search query text.
            k: Number of results.
            filter_dict: Optional metadata filters.

        Returns:
            List of Document objects ordered by relevance.
        """

    @abstractmethod
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
            filter_dict: Optional metadata filters.

        Returns:
            List of (Document, score) tuples ordered by relevance.
        """

    @abstractmethod
    def delete_documents(
        self,
        ids: Optional[list[str]] = None,
        filter_dict: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """Delete documents by IDs or filters.

        Args:
            ids: Optional list of document IDs to delete.
            filter_dict: Optional metadata filters.

        Returns:
            Number of documents deleted.
        """

    @abstractmethod
    def get_document_count(self, collection_name: Optional[str] = None) -> int:
        """Return the number of stored documents.

        Args:
            collection_name: Optional collection to count in.

        Returns:
            Total document count.
        """

    @abstractmethod
    def reset(self, collection_name: Optional[str] = None) -> None:
        """Delete all data in the store.

        Args:
            collection_name: Optional collection to reset.
        """

    @abstractmethod
    def get_collection_names(self) -> list[str]:
        """Return all available collection names.

        Returns:
            List of collection name strings.
        """

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Hybrid search combining keyword and semantic methods.

        Default implementation falls back to similarity_search_with_score.
        Override in subclasses with native hybrid support.

        Args:
            query: Search query text.
            k: Number of results.
            filter_dict: Optional metadata filters.

        Returns:
            List of (Document, score) tuples.
        """
        return self.similarity_search_with_score(
            query, k=k, filter_dict=filter_dict, collection_name=collection_name
        )

    def save(self) -> None:
        """Persist the vector store to disk.

        Default implementation is a no-op for stores that persist automatically
        (Chroma, Qdrant, Pinecone). Override for FAISS or other stateful stores.
        """

    def load(self) -> None:
        """Load persisted data from disk.

        Default implementation is a no-op. Override for FAISS or other
        stateful stores.
        """

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "BaseVectorStore":
        """Create a store from raw texts (LangChain VectorStore interface).

        Args:
            texts: List of text strings.
            embedding: Embedding function (unused, kept for interface compat).
            metadatas: Optional metadata per text.

        Returns:
            Populated store instance.
        """
        store = cls(embedding_function=embedding, **kwargs)
        docs = [
            Document(page_content=t, metadata=metadatas[i] if metadatas else {})
            for i, t in enumerate(texts)
        ]
        store.add_documents(docs)
        return store

    def list_documents(self) -> list[dict[str, Any]]:
        """List all unique documents in the store.

        Returns a list of dicts with keys: document_id, filename, total_chunks,
        total_pages. Default implementation returns empty list; override in
        subclasses that maintain a docstore.
        """
        return []

    def get_document_content(self, document_id: str) -> list[Document]:
        """Get all chunks for a specific document.

        Args:
            document_id: The document ID to retrieve.

        Returns:
            List of Document objects belonging to that document.
        """
        return []
