"""Reranker for re-ordering retrieved documents by relevance.

Uses a cross-encoder model from ``sentence-transformers`` to score
 ``(query, document)`` pairs and re-sort the results.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.config.settings import Settings, get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Reranker:
    """Cross-encoder based reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialise the reranker.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
            settings: Optional settings override.
        """
        if settings is None:
            settings = get_settings()

        self._model_name = model_name or settings.reranker_model
        # Use the same device as embeddings.  Left unset, sentence-transformers
        # auto-selects MPS on Apple Silicon, where CrossEncoder init can hang.
        self._device = settings.embedding_device
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Lazily load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)
            logger.info("reranker_loaded", model=self._model_name, device=self._device)
        except Exception as exc:
            logger.error(
                "failed_to_load_reranker",
                model=self._model_name,
                error=str(exc),
            )
            self._model = None

    @property
    def is_available(self) -> bool:
        """Check if the reranker model loaded successfully."""
        return self._model is not None

    def rerank(
        self,
        query: str,
        results: list[tuple[Document, float]],
        top_k: int = 3,
    ) -> list[tuple[Document, float]]:
        """Rerank documents using a cross-encoder model.

        Args:
            query: The original query.
            results: Initial retrieval results ``(doc, score)``.
            top_k: Number of top results to return after reranking.

        Returns:
            Re-ranked list of ``(Document, score)`` tuples.
        """
        if self._model is None or not results:
            return results[:top_k]

        pairs = [(query, doc.page_content) for doc, _ in results]
        try:
            scores = self._model.predict(pairs)
            scored = list(zip([doc for doc, _ in results], scores.tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked = [(doc, float(score)) for doc, score in scored[:top_k]]
            logger.info(
                "reranking_complete",
                original_count=len(results),
                reranked_count=len(reranked),
            )
            return reranked
        except Exception as exc:
            logger.error("reranking_failed", error=str(exc))
            return results[:top_k]
