"""Multi-stage parallel retrieval pipeline.

Stage 0: Query Preprocessing
Stage 1: Query Classification
Stage 2: Parallel Retrieval (similarity + hybrid)
Stage 3: Merge + Deduplicate + Normalize
Stage 4: Retrieval Confidence
Stage 5: Reranking (optional)
Stage 6: Context Building
Stage 7: LLM Generation (in rag_service)
Stage 8: Answer Validation (in rag_service)
"""

from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.config.constants import CONDENSE_QUESTION_PROMPT
from app.config.settings import Settings, get_settings
from app.retriever.classifier import ClassifiedQuery, QueryClassifier, QueryType
from app.retriever.confidence import (
    RetrievalConfidence,
    RetrievalResult,
    compute_retrieval_confidence,
    deduplicate_results,
    normalize_scores,
)
from app.retriever.context_builder import BuiltContext, ContextBuilder
from app.retriever.preprocessing import PreprocessedQuery, QueryPreprocessor
from app.utils.helpers import estimate_tokens
from app.utils.logging import get_logger
from app.vectorstore.base import BaseVectorStore

logger = get_logger(__name__)


class RetrievalPipeline:
    """Multi-stage production retrieval pipeline."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialise the pipeline.

        Args:
            vector_store: Vector store for all retrieval strategies.
            llm: LLM used for condensing follow-up questions.
            settings: Optional settings override.
        """
        self._vector_store = vector_store
        self._llm = llm
        self._settings = settings or get_settings()

        # Sub-components
        self._preprocessor = QueryPreprocessor()
        self._classifier = QueryClassifier()
        self._context_builder = ContextBuilder(
            max_tokens=self._settings.max_context_tokens,
        )

    def retrieve(
        self,
        query: str,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        enable_reranking: bool | None = None,
        reranker: Any | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> tuple[list[RetrievalResult], RetrievalConfidence, ClassifiedQuery]:
        """Execute the full retrieval pipeline.

        Args:
            query: User query.
            score_threshold: Minimum score threshold.
            metadata_filter: Optional metadata filter.
            enable_reranking: Override for reranking toggle.
            reranker: Optional reranker instance.
            chat_history: Optional conversation history.

        Returns:
            Tuple of (results, confidence, classification).
        """
        pipeline_start = time.perf_counter()

        # ── Stage 0: Query Preprocessing ─────────────────────────────────
        preprocessed = self._preprocessor.preprocess(query, chat_history)
        search_query = preprocessed.normalized
        extracted_filters = preprocessed.metadata_filters
        if metadata_filter:
            extracted_filters.update(metadata_filter)

        logger.info(
            "stage_0_preprocessing",
            original_length=len(query),
            normalized_length=len(search_query),
            metadata_filters=bool(extracted_filters),
            language=preprocessed.language,
        )

        # ── Stage 1: Query Classification ────────────────────────────────
        classification = self._classifier.classify(search_query)

        logger.info(
            "stage_1_classification",
            query_type=classification.query_type.value,
            confidence=round(classification.confidence, 3),
            suggested_strategies=classification.suggested_strategies,
        )

        # ── Stage 2: Parallel Retrieval ──────────────────────────────────
        stage2_start = time.perf_counter()
        all_results = self._parallel_retrieve(
            search_query, extracted_filters, classification,
        )
        stage2_ms = (time.perf_counter() - stage2_start) * 1000

        # ── Stage 3: Merge + Deduplicate + Normalize ─────────────────────
        deduped = deduplicate_results(all_results)
        normalized = normalize_scores(deduped)

        # Apply score threshold
        threshold = score_threshold if score_threshold is not None else self._settings.score_threshold
        if threshold is not None:
            normalized = [r for r in normalized if r.score >= threshold]

        logger.info(
            "stage_3_merge",
            total_raw=len(all_results),
            after_dedup=len(deduped),
            after_threshold=len(normalized),
        )

        # ── Stage 4: Retrieval Confidence ────────────────────────────────
        confidence = compute_retrieval_confidence(normalized)

        logger.info(
            "stage_4_confidence",
            score=round(confidence.score, 4),
            agreement=round(confidence.agreement_ratio, 3),
            diversity=round(confidence.document_diversity, 3),
            chunk_overlap=round(confidence.chunk_overlap_ratio, 3),
        )

        # ── Stage 5: Reranking (optional) ────────────────────────────────
        should_rerank = (
            enable_reranking
            if enable_reranking is not None
            else self._settings.enable_reranking
        )
        if should_rerank and reranker is not None and normalized:
            rk = self._settings.rerank_top_k
            reranked = reranker.rerank(
                search_query,
                [(r.document, r.score) for r in normalized],
                top_k=rk,
            )
            normalized = [
                RetrievalResult(document=doc, score=score, strategy="reranked")
                for doc, score in reranked
            ]
            logger.info(
                "stage_5_reranking",
                before=len(deduped),
                after=len(normalized),
            )

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            "pipeline_stages_complete",
            final_results=len(normalized),
            confidence=round(confidence.score, 4),
            total_latency_ms=round(total_ms, 2),
        )

        return normalized, confidence, classification

    def build_context(
        self,
        results: list[RetrievalResult],
        max_tokens: int | None = None,
    ) -> BuiltContext:
        """Build final context for the LLM (Stage 7).

        Args:
            results: Ranked retrieval results.
            max_tokens: Override for token budget.

        Returns:
            BuiltContext with selected results and statistics.
        """
        return self._context_builder.build(results, max_tokens)

    def _parallel_retrieve(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None,
        classification: ClassifiedQuery,
    ) -> list[RetrievalResult]:
        """Run retrieval strategies in parallel (Stage 2).

        Args:
            query: Search query.
            metadata_filter: Optional metadata filter.
            classification: Query classification result.

        Returns:
            Combined results from all strategies.
        """
        # Strategies run sequentially, not in parallel threads: similarity and
        # hybrid both embed the query with the same HuggingFace fast tokenizer,
        # which is not thread-safe (concurrent use panics with "Already
        # borrowed"). Hybrid is dominated by CPU-bound BM25 work anyway, so
        # parallelism bought little.
        strategies: list[tuple[str, Any, int]] = [
            ("similarity", self._similarity_search, self._settings.similarity_top_k),
        ]
        if self._settings.enable_hybrid_search:
            strategies.append(
                ("hybrid", self._hybrid_search, self._settings.hybrid_top_k)
            )

        all_results: list[RetrievalResult] = []
        for name, search_fn, k in strategies:
            try:
                all_results.extend(search_fn(query, k, metadata_filter))
            except Exception as exc:
                logger.error("retrieval_strategy_failed", strategy=name, error=str(exc))

        return all_results

    def _similarity_search(
        self, query: str, k: int, metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        """Execute similarity search."""
        start = time.perf_counter()
        results = self._vector_store.similarity_search_with_score(
            query, k=k, filter_dict=metadata_filter,
        )
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("similarity_complete", count=len(results), ms=round(elapsed, 2))
        return [
            RetrievalResult(document=doc, score=score, strategy="similarity")
            for doc, score in results
        ]

    def _hybrid_search(
        self, query: str, k: int, metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        """Execute hybrid search."""
        start = time.perf_counter()
        results = self._vector_store.hybrid_search(
            query, k=k, filter_dict=metadata_filter,
        )
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("hybrid_complete", count=len(results), ms=round(elapsed, 2))
        return [
            RetrievalResult(document=doc, score=score, strategy="hybrid")
            for doc, score in results
        ]

    def select_final_context(
        self, results: list[RetrievalResult], max_tokens: int | None = None,
    ) -> list[RetrievalResult]:
        """Select final context within token budget (Stage 7).

        Args:
            results: Ranked results.
            max_tokens: Token budget override.

        Returns:
            Selected results.
        """
        built = self._context_builder.build(results, max_tokens)
        return built.results

    def condense_question(
        self,
        chat_history: list[dict[str, str]],
        question: str,
    ) -> str:
        """Rephrase a follow-up question as standalone."""
        if not chat_history or self._llm is None:
            return question

        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in chat_history[-self._settings.memory_window :]
        )
        prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=history_text, question=question,
        )
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            return (response.content if hasattr(response, "content") else str(response)).strip()
        except Exception as exc:
            logger.warning("condense_failed", error=str(exc))
            return question


# ── Legacy compatibility ──────────────────────────────────────────────────────


class RAGRetriever:
    """Legacy wrapper for backward compatibility."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._pipeline = RetrievalPipeline(vector_store, llm, settings)
        self._vector_store = vector_store
        self._llm = llm
        self._settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        retrieval_mode: str = "similarity",
        metadata_filter: dict[str, Any] | None = None,
        enable_reranking: bool = False,
        reranker: Any | None = None,
        rerank_top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Legacy retrieve interface."""
        results, confidence, cls = self._pipeline.retrieve(
            query=query,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
            enable_reranking=enable_reranking,
            reranker=reranker,
        )
        if top_k is not None:
            results = results[:top_k]
        return [(r.document, r.score) for r in results]

    def condense_question(
        self, chat_history: list[dict[str, str]], question: str,
    ) -> str:
        return self._pipeline.condense_question(chat_history, question)

    def build_rag_chain(self) -> Any:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from app.config.constants import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE

        if self._llm is None:
            raise ValueError("LLM is required to build a RAG chain")

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT_TEMPLATE),
        ])

        def _format_docs(docs: list[Document]) -> str:
            return "\n\n".join(
                f"[Source {i+1}: {d.metadata.get('filename', 'unknown')}, "
                f"Page {d.metadata.get('page_number', '?')}]\n{d.page_content}"
                for i, d in enumerate(docs)
            )

        def _retrieve(query: str) -> list[Document]:
            results, _, _ = self._pipeline.retrieve(query)
            return [r.document for r in results]

        return (
            {"context": _retrieve | _format_docs, "question": RunnablePassthrough()}
            | prompt | self._llm | StrOutputParser()
        )
