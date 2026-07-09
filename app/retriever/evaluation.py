"""Retrieval evaluation metrics for RAG pipeline assessment.

Implements standard information retrieval metrics: Recall@k, Precision@k,
MRR, nDCG, Hit Rate, and coverage analysis.  Supports batch evaluation
and exportable reports.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document

from app.retriever.confidence import RetrievalResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Single query evaluation result."""

    query: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    latency_ms: float
    num_retrieved: int
    num_relevant: int
    relevant_doc_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across multiple queries."""

    total_queries: int
    avg_recall_at_k: float
    avg_precision_at_k: float
    avg_mrr: float
    avg_ndcg_at_k: float
    avg_hit_rate: float
    avg_latency_ms: float
    query_results: list[EvaluationResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall@k: fraction of relevant docs retrieved in top-k.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall score between 0 and 1.
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Precision@k: fraction of top-k retrieved docs that are relevant.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Precision score between 0 and 1.
    """
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_count = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return relevant_count / k


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Compute MRR: reciprocal rank of the first relevant document.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of ground-truth relevant document IDs.

    Returns:
        MRR score between 0 and 1.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute nDCG@k: normalized discounted cumulative gain.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        nDCG score between 0 and 1.
    """
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Hit Rate@k: whether any relevant doc is in top-k.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        1.0 if at least one relevant doc in top-k, else 0.0.
    """
    retrieved_k = set(retrieved[:k])
    return 1.0 if retrieved_k & relevant else 0.0


def evaluate_retrieval(
    query: str,
    results: list[RetrievalResult],
    relevant_doc_ids: set[str],
    k: int = 5,
    latency_ms: float = 0.0,
) -> EvaluationResult:
    """Evaluate retrieval results for a single query.

    Args:
        query: The original query string.
        results: Ranked retrieval results.
        relevant_doc_ids: Ground-truth relevant document IDs.
        k: Evaluation cutoff.
        latency_ms: Retrieval latency in milliseconds.

    Returns:
        EvaluationResult with all metrics.
    """
    retrieved_ids = [
        r.document.metadata.get("document_id", r.document.metadata.get("source", ""))
        for r in results
    ]

    return EvaluationResult(
        query=query,
        recall_at_k=recall_at_k(retrieved_ids, relevant_doc_ids, k),
        precision_at_k=precision_at_k(retrieved_ids, relevant_doc_ids, k),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant_doc_ids),
        ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_doc_ids, k),
        hit_rate=hit_rate_at_k(retrieved_ids, relevant_doc_ids, k),
        latency_ms=latency_ms,
        num_retrieved=len(results),
        num_relevant=len(relevant_doc_ids),
        relevant_doc_ids=list(relevant_doc_ids),
        retrieved_doc_ids=retrieved_ids,
    )


def evaluate_batch(
    queries: list[dict[str, Any]],
    retriever_fn: Any,
    k: int = 5,
) -> EvaluationReport:
    """Evaluate retrieval across multiple queries.

    Args:
        queries: List of dicts with 'query' and 'relevant_doc_ids' keys.
        retriever_fn: Callable that takes (query, k) and returns list[RetrievalResult].
        k: Evaluation cutoff.

    Returns:
        Aggregated EvaluationReport.
    """
    results: list[EvaluationResult] = []

    for item in queries:
        query = item["query"]
        relevant = set(item.get("relevant_doc_ids", []))

        start = time.perf_counter()
        retrieved = retriever_fn(query, k)
        latency = (time.perf_counter() - start) * 1000

        eval_result = evaluate_retrieval(
            query=query,
            results=retrieved,
            relevant_doc_ids=relevant,
            k=k,
            latency_ms=latency,
        )
        results.append(eval_result)

    n = len(results) or 1
    report = EvaluationReport(
        total_queries=len(results),
        avg_recall_at_k=sum(r.recall_at_k for r in results) / n,
        avg_precision_at_k=sum(r.precision_at_k for r in results) / n,
        avg_mrr=sum(r.mrr for r in results) / n,
        avg_ndcg_at_k=sum(r.ndcg_at_k for r in results) / n,
        avg_hit_rate=sum(r.hit_rate for r in results) / n,
        avg_latency_ms=sum(r.latency_ms for r in results) / n,
        query_results=results,
        summary={
            "total_queries": len(results),
            "k": k,
            "metrics": {
                "recall_at_k": round(sum(r.recall_at_k for r in results) / n, 4),
                "precision_at_k": round(sum(r.precision_at_k for r in results) / n, 4),
                "mrr": round(sum(r.mrr for r in results) / n, 4),
                "ndcg_at_k": round(sum(r.ndcg_at_k for r in results) / n, 4),
                "hit_rate": round(sum(r.hit_rate for r in results) / n, 4),
                "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 2),
            },
        },
    )

    logger.info(
        "evaluation_complete",
        total_queries=report.total_queries,
        recall=report.summary["metrics"]["recall_at_k"],
        precision=report.summary["metrics"]["precision_at_k"],
        mrr=report.summary["metrics"]["mrr"],
        ndcg=report.summary["metrics"]["ndcg_at_k"],
        hit_rate=report.summary["metrics"]["hit_rate"],
    )

    return report
