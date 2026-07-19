"""FAISS vector store implementation.

FAISS is a high-performance library for similarity search.
This implementation adds persistence (index + metadata), metadata filtering,
MMR diversity, and a hybrid search fallback.

Usage:
    store = FAISSVectorStore(embedding_fn)
    store.add_documents(docs)
    store.save()                    # persists index + metadata to disk
    results = store.similarity_search("query", k=5)
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document

from app.utils.logging import get_logger
from app.vectorstore.base import BaseVectorStore

logger = get_logger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed vector store with persistence and metadata filtering."""

    def __init__(
        self,
        embedding_function: Any,
        persist_directory: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._embedding_function = embedding_function
        self._persist_directory = Path(persist_directory) if persist_directory else None
        self._index = None
        self._metadata: list[dict[str, Any]] = []
        self._docstore: dict[str, Document] = {}
        self._ids: list[str] = []
        self._collection_name = kwargs.get("collection_name", "default")
        self._bm25 = None
        self._bm25_dirty = True
        self._lock = threading.Lock()
        self._build_index()
        self._auto_load()

    def _auto_load(self) -> None:
        """Auto-load persisted index if it exists."""
        if self._persist_directory is None:
            return
        index_path = self._persist_directory / "index.faiss"
        if index_path.exists():
            try:
                self.load()
                logger.info(
                    "faiss_auto_loaded",
                    path=str(self._persist_directory),
                    doc_count=len(self._docstore),
                )
            except Exception as exc:
                logger.warning("faiss_auto_load_failed", error=str(exc))

    def _build_index(self) -> None:
        """Import FAISS lazily. The index itself is created on first use,
        once the true embedding dimension is known from actual vectors."""
        try:
            import faiss
            import numpy as np

            # Pin FAISS to a single OpenMP thread. When torch/sentence-transformers
            # is also loaded (as it always is here), FAISS's multithreaded search
            # can deadlock against torch's OpenMP runtime on macOS. Single-threaded
            # search over these index sizes is still sub-millisecond.
            try:
                faiss.omp_set_num_threads(1)
            except Exception:  # pragma: no cover - defensive
                pass

            self._faiss = faiss
            self._np = np
            self._index = None
            self._dimension: Optional[int] = None
        except ImportError:
            raise ImportError(
                "FAISS is required for FAISSVectorStore. "
                "Install with: pip install faiss-cpu or faiss-gpu"
            )

    def _embed_texts(self, texts: list[str]) -> Any:
        """Embed a list of texts into a numpy array."""
        embeddings = self._embedding_function.embed_documents(texts)
        return self._np.array(embeddings, dtype=self._np.float32)

    def _ensure_index(self, dim: int) -> None:
        """Create the FAISS index on first use with the observed dimension."""
        if self._index is None:
            self._index = self._faiss.IndexFlatL2(dim)
            self._dimension = dim

    def _refresh_index(self) -> None:
        """Rebuild the FAISS index from the current docstore."""
        if self._docstore:
            texts = [d.page_content for d in self._docstore.values()]
            vectors = self._embed_texts(texts)
            self._index = self._faiss.IndexFlatL2(int(vectors.shape[1]))
            self._dimension = int(vectors.shape[1])
            self._index.add(vectors)
        elif self._dimension is not None:
            self._index = self._faiss.IndexFlatL2(self._dimension)
        else:
            self._index = None

    # ---- BaseVectorStore interface ----

    def add_documents(
        self,
        documents: list[Document],
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents, embed, and store in the FAISS index.

        Args:
            documents: LangChain Document objects.
            collection_name: Unused (single-collection store).

        Returns:
            List of generated document IDs.
        """
        if not documents:
            return []

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        ids = [kwargs.get("ids", [None] * len(documents))[i] or f"doc_{hash(texts[i])}" for i in range(len(documents))]

        vectors = self._embed_texts(texts)

        with self._lock:
            self._ensure_index(int(vectors.shape[1]))
            self._index.add(vectors)

            for i, doc in enumerate(documents):
                doc_id = ids[i]
                self._docstore[doc_id] = doc
                self._metadata.append(
                    {"id": doc_id, "metadata": metadatas[i], "text_preview": texts[i][:100]}
                )
                self._ids.append(doc_id)

            logger.info("faiss_documents_added", count=len(documents))
            self._bm25_dirty = True
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
            filter_dict: Optional metadata filter {key: value}.

        Returns:
            List of Document objects ordered by relevance.
        """
        results = self.similarity_search_with_score(
            query, k=k * 3 if filter_dict else k, filter_dict=filter_dict
        )
        return [doc for doc, _ in results[:k]]

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
            filter_dict: Optional metadata filter {key: value}.

        Returns:
            List of (Document, score) tuples.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query_vector = self._embed_texts([query])
        actual_k = min(k, self._index.ntotal)

        with self._lock:
            distances, indices = self._index.search(query_vector, actual_k)

        results: list[tuple[Document, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self._ids):
                continue
            doc_id = self._ids[idx]
            doc = self._docstore.get(doc_id)
            if doc is None:
                continue
            if filter_dict and not self._matches_filter(doc.metadata, filter_dict):
                continue
            results.append((doc, float(dist)))
            if len(results) >= k:
                break

        return results

    def _get_bm25(self) -> Any:
        """Get or build the BM25 index lazily."""
        if self._bm25 is None or self._bm25_dirty:
            from app.retriever.bm25_search import BM25Search

            self._bm25 = BM25Search()
            docs = list(self._docstore.values())
            if docs:
                self._bm25.build_index(docs)
            self._bm25_dirty = False
        return self._bm25

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Hybrid search combining BM25 lexical + dense vector via RRF.

        Overrides the base fallback to provide true hybrid retrieval.

        Args:
            query: Search query text.
            k: Number of results.
            filter_dict: Optional metadata filters.

        Returns:
            List of (Document, score) tuples ranked by RRF fusion.
        """
        from app.retriever.bm25_search import reciprocal_rank_fusion

        # Dense vector search
        dense_results = self.similarity_search_with_score(
            query, k=k * 3, filter_dict=filter_dict,
        )

        # BM25 lexical search
        bm25 = self._get_bm25()
        bm25_results = bm25.search(query, k=k * 3)

        # Apply metadata filter to BM25 results if specified
        if filter_dict:
            bm25_results = [
                (doc, score) for doc, score in bm25_results
                if self._matches_filter(doc.metadata, filter_dict)
            ]

        # Combine via Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion([dense_results, bm25_results])

        logger.debug(
            "hybrid_search_complete",
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
            fused_count=len(fused),
        )
        return fused[:k]

    def delete_documents(
        self,
        ids: Optional[list[str]] = None,
        filter_dict: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """Delete documents by IDs or rebuild with filter applied.

        Args:
            ids: Optional list of document IDs to delete.
            filter_dict: Optional metadata filter.

        Returns:
            Number of documents deleted.
        """
        with self._lock:
            if ids:
                removed = sum(1 for i in ids if self._docstore.pop(i, None) is not None)
            elif filter_dict:
                removed = 0
                for doc_id in list(self._docstore):
                    if self._matches_filter(self._docstore[doc_id].metadata, filter_dict):
                        self._docstore.pop(doc_id)
                        removed += 1
            else:
                removed = len(self._docstore)
                self._docstore.clear()

            if removed:
                self._refresh_index()
                self._metadata = [m for m in self._metadata if m["id"] in self._docstore]
                self._ids = [i for i in self._ids if i in self._docstore]
                self._bm25_dirty = True

        return removed

    def get_document_count(self, collection_name: Optional[str] = None) -> int:
        """Return the number of stored documents."""
        return len(self._docstore)

    def reset(self, collection_name: Optional[str] = None) -> None:
        """Delete all data and rebuild a blank index."""
        self._docstore.clear()
        self._metadata.clear()
        self._ids.clear()
        self._bm25_dirty = True
        self._refresh_index()
        logger.info("faiss_store_reset")

    def get_collection_names(self) -> list[str]:
        """Return the single collection name."""
        return [self._collection_name]

    def save(self, path: Optional[str] = None, **kwargs: Any) -> None:
        """Persist index and metadata to disk.

        Args:
            path: Optional directory override.  Falls back to ``persist_directory``.
        """
        dir_path = Path(path) if path else self._persist_directory
        if dir_path is None:
            logger.warning("faiss_save_no_path", msg="No persist directory set; skipping save.")
            return

        with self._lock:
            dir_path.mkdir(parents=True, exist_ok=True)
            if self._index is not None:
                self._faiss.write_index(self._index, str(dir_path / "index.faiss"))
            (dir_path / "metadata.json").write_text(json.dumps(self._metadata, indent=2))
            (dir_path / "ids.json").write_text(json.dumps(self._ids))
            docstore_data = {k: {"page_content": v.page_content, "metadata": v.metadata} for k, v in self._docstore.items()}
            (dir_path / "docstore.json").write_text(json.dumps(docstore_data))
            logger.info("faiss_saved", path=str(dir_path), doc_count=len(self._docstore))

    def load(self, path: Optional[str] = None, **kwargs: Any) -> None:
        """Load persisted index and metadata from disk.

        Args:
            path: Optional directory override.
        """
        dir_path = Path(path) if path else self._persist_directory
        if dir_path is None or not dir_path.exists():
            logger.warning("faiss_load_missing", path=str(dir_path))
            return

        index_path = dir_path / "index.faiss"
        if index_path.exists():
            self._index = self._faiss.read_index(str(index_path))
            self._dimension = int(self._index.d)
        meta_path = dir_path / "metadata.json"
        if meta_path.exists():
            self._metadata = json.loads(meta_path.read_text())
        ids_path = dir_path / "ids.json"
        if ids_path.exists():
            self._ids = json.loads(ids_path.read_text())
        docstore_path = dir_path / "docstore.json"
        if docstore_path.exists():
            raw = json.loads(docstore_path.read_text())
            self._docstore = {
                k: Document(page_content=v["page_content"], metadata=v["metadata"])
                for k, v in raw.items()
            }
        logger.info("faiss_loaded", path=str(dir_path), doc_count=len(self._docstore))

    def list_documents(self) -> list[dict[str, Any]]:
        """List all unique documents in the store.

        Returns a list of dicts with keys: document_id, filename, total_chunks,
        total_pages, created_at.
        """
        doc_groups: dict[str, dict[str, Any]] = {}

        for doc in self._docstore.values():
            meta = doc.metadata
            doc_id = meta.get("document_id", "unknown")
            if doc_id not in doc_groups:
                doc_groups[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "total_chunks": 0,
                    "total_pages": set(),
                    "created_at": meta.get("created_at"),
                }
            doc_groups[doc_id]["total_chunks"] += 1
            page = meta.get("page_number")
            if page is not None:
                doc_groups[doc_id]["total_pages"].add(page)

        result = []
        for doc_id, info in doc_groups.items():
            info["total_pages"] = len(info["total_pages"]) if info["total_pages"] else 0
            result.append(info)

        return sorted(result, key=lambda x: x.get("created_at") or "", reverse=True)

    def get_document_content(self, document_id: str) -> list[Document]:
        """Get all chunks for a specific document.

        Args:
            document_id: The document ID to retrieve.

        Returns:
            List of Document objects belonging to that document, sorted by page/chunk.
        """
        chunks = []
        for doc in self._docstore.values():
            if doc.metadata.get("document_id") == document_id:
                chunks.append(doc)

        # Sort by page number then chunk_id
        def sort_key(d: Document) -> tuple[int, int]:
            return (
                d.metadata.get("page_number", 0),
                d.metadata.get("chunk_id", 0),
            )

        return sorted(chunks, key=sort_key)

    # ---- Private helpers ----

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
        """Check if all filter key-value pairs exist in metadata."""
        return all(metadata.get(k) == v for k, v in filter_dict.items())
