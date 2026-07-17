"""Core RAG service — the central orchestrator.

Coordinates upload, ingestion, multi-stage retrieval, generation, chat,
deletion, statistics, feedback, and cost estimation.  All components are
lazy-loaded via properties.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.config.constants import RAG_SYSTEM_PROMPT
from app.config.settings import ChunkingStrategy, Settings, get_settings
from app.embeddings.factory import get_embeddings_model
from app.llm.factory import get_llm
from app.models.schemas import (
    ChatRequest,
    HealthResponse,
    IngestResponse,
    IngestedDocument,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    StatsResponse,
    StreamChunk,
)
from app.retriever.confidence import (
    AnswerConfidence,
    RetrievalConfidence,
    RetrievalResult,
    compute_answer_confidence,
)
from app.retriever.rag_retriever import RAGRetriever, RetrievalPipeline
from app.retriever.reranker import Reranker
from app.retriever.validator import AnswerValidator, ValidationResult
from app.utils.helpers import estimate_tokens, generate_id
from app.utils.logging import get_logger
from app.vectorstore.base import BaseVectorStore
from app.vectorstore.factory import get_vector_store

logger = get_logger(__name__)

_MAX_SESSIONS = 200


class RAGService:
    """Singleton-style service managing the full RAG lifecycle."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise all components.

        Args:
            settings: Optional settings override.
        """
        self._settings = settings or get_settings()
        self._vector_store: Optional[BaseVectorStore] = None
        self._pipeline: Optional[RetrievalPipeline] = None
        self._retriever: Optional[RAGRetriever] = None
        self._llm = None
        self._reranker: Optional[Reranker] = None
        self._validator: Optional[AnswerValidator] = None
        self._embeddings = None

        self._total_queries: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._total_latency: float = 0.0

        self._sessions: dict[str, list[dict[str, str]]] = {}

    def refresh_settings(self, db_overrides: dict[str, str] | None = None) -> None:
        """Refresh settings from the environment and optional DB overrides.

        Call this after settings are updated via the API to pick up changes.
        Resets cached components that depend on settings.

        Args:
            db_overrides: Optional dict of key-value overrides from the database
                settings API. These take precedence over env vars.
        """
        from app.config.settings import get_settings

        self._settings = get_settings()

        # Apply DB overrides to the settings object
        if db_overrides:
            self._apply_db_overrides(db_overrides)

        # Reset cached components that depend on settings
        self._pipeline = None
        self._retriever = None
        self._reranker = None
        self._validator = None
        logger.info("rag_service_settings_refreshed", overrides=list(db_overrides or []))

    def _apply_db_overrides(self, overrides: dict[str, str]) -> None:
        """Apply database setting overrides to the settings object.

        Args:
            overrides: Dict of setting key-value pairs from the database.
        """
        bool_keys = {
            "enable_hybrid_search",
            "enable_reranking", "enable_answer_validation",
        }
        int_keys = {
            "similarity_top_k", "hybrid_top_k",
            "chunk_size", "chunk_overlap", "max_context_tokens", "rerank_top_k",
        }
        float_keys = {"score_threshold", "temperature"}
        str_keys = {
            "llm_provider", "openai_model", "anthropic_model",
            "openrouter_model", "embedding_model", "vector_store",
        }

        for key, value in overrides.items():
            if key in bool_keys:
                setattr(self._settings, key, value.lower() in ("true", "1", "yes"))
            elif key in int_keys:
                try:
                    setattr(self._settings, key, int(value))
                except (ValueError, TypeError):
                    pass
            elif key in float_keys:
                try:
                    setattr(self._settings, key, float(value))
                except (ValueError, TypeError):
                    pass
            elif key in str_keys:
                # Some str_keys are enums, handle them
                if key == "llm_provider":
                    try:
                        from app.config.settings import LLMProvider
                        self._settings.llm_provider = LLMProvider(value)
                    except (ValueError, TypeError):
                        pass
                elif key == "embedding_model":
                    try:
                        from app.config.settings import EmbeddingModelType
                        self._settings.embedding_model = EmbeddingModelType(value)
                    except (ValueError, TypeError):
                        pass
                elif key == "vector_store":
                    try:
                        from app.config.settings import VectorStoreType
                        self._settings.vector_store = VectorStoreType(value)
                    except (ValueError, TypeError):
                        pass
                else:
                    setattr(self._settings, key, value)

    @property
    def vector_store(self) -> BaseVectorStore:
        """Lazy-load the vector store."""
        if self._vector_store is None:
            self._embeddings = get_embeddings_model(self._settings)
            self._vector_store = get_vector_store(self._embeddings, self._settings)
        return self._vector_store

    @property
    def llm(self):  # noqa: ANN202
        """Lazy-load the LLM."""
        if self._llm is None:
            self._llm = get_llm(self._settings)
        return self._llm

    @property
    def pipeline(self) -> RetrievalPipeline:
        """Lazy-load the multi-stage retrieval pipeline."""
        if self._pipeline is None:
            self._pipeline = RetrievalPipeline(
                vector_store=self.vector_store,
                llm=self.llm,
                settings=self._settings,
            )
        return self._pipeline

    @property
    def retriever(self) -> RAGRetriever:
        """Lazy-load the legacy retriever wrapper."""
        if self._retriever is None:
            self._retriever = RAGRetriever(
                vector_store=self.vector_store,
                llm=self.llm,
                settings=self._settings,
            )
        return self._retriever

    @property
    def reranker(self) -> Reranker | None:
        """Lazy-load the reranker if enabled."""
        if self._settings.enable_reranking and self._reranker is None:
            self._reranker = Reranker(settings=self._settings)
        return self._reranker

    @property
    def validator(self) -> AnswerValidator | None:
        """Lazy-load the answer validator if enabled."""
        if self._settings.enable_answer_validation and self._validator is None:
            self._validator = AnswerValidator(llm=self.llm)
        return self._validator

    def query(self, request: QueryRequest) -> QueryResponse:
        """Execute a RAG query through the full pipeline.

        Stage 0: Preprocessing → Stage 1: Classification → Stage 2: Parallel
        retrieval → Stage 3: Merge/dedup → Stage 4: Confidence → Stage 5:
        Reranking → Stage 6: Context building → Stage 7: Generation →
        Stage 8: Validation.

        Args:
            request: ``QueryRequest`` with question and parameters.

        Returns:
            ``QueryResponse`` with answer, sources, and metadata.
        """
        start = time.perf_counter()
        query_id = generate_id()

        logger.info("processing_query", query_id=query_id, question_length=len(request.question))

        # ── Stages 0-7: Multi-stage retrieval pipeline ────────────────────
        results, retrieval_confidence, classification = self.pipeline.retrieve(
            query=request.question,
            score_threshold=request.score_threshold,
            metadata_filter=request.metadata_filter,
            enable_reranking=(
                request.enable_reranking
                if request.enable_reranking is not None
                else self._settings.enable_reranking
            ),
            reranker=self.reranker,
        )

        # ── Stage 7: Context selection with token budget ──────────────────
        context_results = self.pipeline.select_final_context(results)

        # ── Stage 8: Generate answer ──────────────────────────────────────
        context = self._format_context(context_results)
        answer = self._generate_answer(context, request.question)

        # ── Stage 9: Answer validation ────────────────────────────────────
        validation_result: ValidationResult | None = None
        if self.validator is not None:
            validation_result = self.validator.validate(
                answer=answer,
                sources=context_results,
                query=request.question,
            )
            logger.info(
                "stage_9_validation",
                is_valid=validation_result.is_valid,
                coverage=round(validation_result.coverage_score, 3),
                unsupported=len(validation_result.unsupported_statements),
            )

        # ── Build sources and compute answer confidence ──────────────────
        sources = self._build_sources(context_results)
        reranker_scores = (
            [r.score for r in context_results if r.strategy == "reranked"]
            if any(r.strategy == "reranked" for r in context_results)
            else None
        )
        # Source agreement: fraction of sources with the same top file
        source_agreement = self._compute_source_agreement(context_results)
        answer_confidence = compute_answer_confidence(
            retrieval_confidence=retrieval_confidence,
            reranker_scores=reranker_scores,
            doc_count=len(context_results),
            source_agreement=source_agreement,
        )

        tokens = estimate_tokens(answer) + sum(estimate_tokens(s.content) for s in sources)
        cost = self._estimate_cost(tokens)
        latency = (time.perf_counter() - start) * 1000

        self._total_queries += 1
        self._total_tokens += tokens
        self._total_cost += cost
        self._total_latency += latency

        logger.info(
            "query_complete",
            query_id=query_id,
            query_type=classification.query_type.value,
            sources_count=len(sources),
            tokens=tokens,
            latency_ms=round(latency, 2),
            confidence=answer_confidence.level,
            confidence_score=round(answer_confidence.score, 4),
            retrieval_confidence=round(retrieval_confidence.score, 4),
            validation_ok=validation_result.is_valid if validation_result else None,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            query_id=query_id,
            confidence_score=round(answer_confidence.score, 4),
            confidence_level=answer_confidence.level,
            model_used=self._settings.get_llm_model_name(),
            tokens_used=tokens,
            cost_estimate=round(cost, 6),
            latency_ms=round(latency, 2),
            query_type=classification.query_type.value,
            validation=validation_result.to_dict() if validation_result else None,
        )

    def chat(self, request: ChatRequest) -> QueryResponse:
        """Execute a conversational chat with memory.

        Args:
            request: ``ChatRequest`` with message and session info.

        Returns:
            ``QueryResponse`` with answer and sources.
        """
        session_id = request.session_id or generate_id()
        if session_id not in self._sessions:
            self._prune_sessions()
            self._sessions[session_id] = []

        history = self._sessions[session_id]

        if self._settings.enable_memory and history:
            standalone_q = self.pipeline.condense_question(history, request.message)
        else:
            standalone_q = request.message

        query_req = QueryRequest(
            question=standalone_q,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            enable_reranking=request.enable_reranking,
        )
        response = self.query(query_req)

        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": response.answer})

        max_messages = self._settings.memory_window * 2
        if len(history) > max_messages:
            self._sessions[session_id] = history[-max_messages:]

        response.session_id = session_id
        return response

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream a chat response token by token.

        Args:
            request: ``ChatRequest`` with ``stream=True``.

        Yields:
            ``StreamChunk`` objects with individual tokens.
        """
        session_id = request.session_id or generate_id()
        query_id = generate_id()

        if session_id not in self._sessions:
            self._prune_sessions()
            self._sessions[session_id] = []

        history = self._sessions[session_id]

        if self._settings.enable_memory and history:
            standalone_q = self.pipeline.condense_question(history, request.message)
        else:
            standalone_q = request.message

        # ── Multi-stage retrieval ────────────────────────────────────────
        results, retrieval_confidence, classification = self.pipeline.retrieve(
            query=standalone_q,
            score_threshold=None,
            metadata_filter=None,
            enable_reranking=(
                request.enable_reranking
                if request.enable_reranking is not None
                else self._settings.enable_reranking
            ),
            reranker=self.reranker,
        )
        context_results = self.pipeline.select_final_context(results)

        # ── Stream answer ────────────────────────────────────────────────
        context = self._format_context(context_results)
        system_content = RAG_SYSTEM_PROMPT
        human_content = (
            f"Context:\n{context}\n\nQuestion: {request.message}\n\n"
            "Provide a detailed answer with citations."
        )

        full_answer = ""
        async for chunk in self.llm.astream([
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer += token
            yield StreamChunk(token=token, query_id=query_id, is_final=False)

        sources = self._build_sources(context_results)

        # ── Post-generation validation ───────────────────────────────────
        validation_result: ValidationResult | None = None
        if self.validator is not None:
            validation_result = self.validator.validate(
                answer=full_answer,
                sources=context_results,
                query=standalone_q,
            )

        yield StreamChunk(
            token="",
            query_id=query_id,
            is_final=True,
            sources=sources,
            metadata={
                "session_id": session_id,
                "model": self._settings.get_llm_model_name(),
                "retrieval_confidence": round(retrieval_confidence.score, 4),
                "query_type": classification.query_type.value,
                "validation": validation_result.to_dict() if validation_result else None,
            },
        )

        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": full_answer})

        max_messages = self._settings.memory_window * 2
        if len(history) > max_messages:
            self._sessions[session_id] = history[-max_messages:]

    def upload_and_ingest(
        self,
        file_paths: list[str],
        chunking_strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestResponse:
        """Upload and ingest files into the vector store.

        Args:
            file_paths: Paths to files.
            chunking_strategy: Strategy override.
            chunk_size: Size override.
            chunk_overlap: Overlap override.

        Returns:
            ``IngestResponse`` with document summaries.
        """
        from app.ingestion.pipeline import ingest_files

        strategy = ChunkingStrategy(chunking_strategy) if chunking_strategy else None

        docs = ingest_files(
            file_paths=file_paths,
            vector_store=self.vector_store,
            settings=self._settings,
            chunking_strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        total_chunks = sum(d.total_chunks for d in docs)
        return IngestResponse(
            documents=docs,
            total_chunks=total_chunks,
            message=f"Successfully ingested {len(docs)} document(s) into {total_chunks} chunks.",
        )

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks from the vector store.

        Args:
            document_id: The document ID to remove.

        Returns:
            ``True`` on success.
        """
        deleted_count = self.vector_store.delete_documents([document_id])
        if deleted_count > 0 and hasattr(self.vector_store, "save"):
            self.vector_store.save()
        return deleted_count > 0

    def get_documents(self) -> list[IngestedDocument]:
        """List all indexed documents.

        Returns:
            List of ``IngestedDocument`` summaries.
        """
        doc_list = self.vector_store.list_documents()
        return [
            IngestedDocument(
                document_id=doc["document_id"],
                filename=doc["filename"],
                total_chunks=doc["total_chunks"],
                total_pages=doc.get("total_pages", 0),
                created_at=doc.get("created_at"),
            )
            for doc in doc_list
        ]

    def get_health(self) -> HealthResponse:
        """Return system health status with component checks.

        Returns:
            HealthResponse with component status.
        """
        from app.config.constants import APP_VERSION

        health = HealthResponse(
            status="healthy",
            version=APP_VERSION,
            vector_store=self._settings.vector_store.value,
            embedding_model=self._settings.embedding_model.value,
            llm_provider=self._settings.llm_provider.value,
        )

        # Check vector store
        try:
            docs = self.vector_store.list_documents()
            health.documents_indexed = len(docs)
        except Exception as exc:
            logger.warning("health_check_vector_store_failed", error=str(exc))

        # Check database connectivity
        try:
            from app.models.database import _get_session_factory
            from sqlalchemy import text
            SessionLocal = _get_session_factory()
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
        except Exception as exc:
            logger.warning("health_check_database_failed", error=str(exc))

        return health

    def get_stats(self) -> StatsResponse:
        """Return system statistics.

        Returns:
            ``StatsResponse`` with usage metrics.
        """
        avg_latency = self._total_latency / max(self._total_queries, 1)
        return StatsResponse(
            total_documents=self.vector_store.get_document_count(),
            total_chunks=self.vector_store.get_document_count(),
            total_queries=self._total_queries,
            total_tokens_used=self._total_tokens,
            total_cost=round(self._total_cost, 6),
            vector_store_type=self._settings.vector_store.value,
            embedding_model=self._settings.embedding_model.value,
            llm_provider=self._settings.llm_provider.value,
            avg_latency_ms=round(avg_latency, 2),
        )

    def _prune_sessions(self) -> None:
        """Evict the oldest sessions when the cap is reached."""
        if len(self._sessions) >= _MAX_SESSIONS:
            oldest_keys = list(self._sessions.keys())[: _MAX_SESSIONS // 2]
            for key in oldest_keys:
                del self._sessions[key]

    def _format_context(
        self,
        results: list[RetrievalResult],
    ) -> str:
        """Format retrieved documents into a context string.

        Args:
            results: List of RetrievalResult objects.

        Returns:
            Formatted context string.
        """
        return "\n\n".join(
            f"[Source {i+1}: {r.document.metadata.get('filename', 'unknown')}, "
            f"Page {r.document.metadata.get('page_number', '?')}, "
            f"Strategy: {r.strategy}]\n{r.document.page_content}"
            for i, r in enumerate(results)
        )

    def _generate_answer(self, context: str, question: str) -> str:
        """Generate an answer using the LLM.

        Args:
            context: Formatted context string.
            question: The user question.

        Returns:
            Generated answer text.
        """
        human_content = (
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Provide a detailed answer with citations."
        )
        response = self.llm.invoke([
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ])
        return response.content if hasattr(response, "content") else str(response)

    def _build_sources(
        self,
        results: list[RetrievalResult],
    ) -> list[SourceDocument]:
        """Build source documents from retrieval results.

        Args:
            results: List of RetrievalResult objects.

        Returns:
            List of ``SourceDocument`` objects.
        """
        return [
            SourceDocument(
                content=r.document.page_content[:500],
                filename=r.document.metadata.get("filename", ""),
                page_number=r.document.metadata.get("page_number", 0),
                chunk_id=r.document.metadata.get("chunk_id", 0),
                document_id=r.document.metadata.get("document_id", ""),
                score=round(r.score, 4),
            )
            for r in results
        ]

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate the cost in USD for a given token count.

        Args:
            tokens: Number of tokens.

        Returns:
            Estimated cost for the active provider (0 for local providers).
        """
        input_cost, _ = self._settings.get_cost_per_1k_tokens()
        return (tokens / 1000) * input_cost

    def _compute_source_agreement(self, results: list[RetrievalResult]) -> float:
        """Fraction of results whose top file matches the majority file.

        Args:
            results: Retrieval results.

        Returns:
            Agreement ratio between 0 and 1.
        """
        if not results:
            return 0.0
        from collections import Counter
        files = [r.document.metadata.get("filename", "") for r in results]
        counts = Counter(files)
        majority_count = counts.most_common(1)[0][1]
        return majority_count / len(files)
