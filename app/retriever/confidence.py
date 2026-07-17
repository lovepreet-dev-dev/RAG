"""Confidence estimation for retrieval and answer quality.

Computes retrieval confidence from multi-strategy results and produces
a final answer confidence combining retrieval, reranking, and LLM signals.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    """A single retrieval result with its score and source strategy."""

    document: Document
    score: float
    strategy: str


@dataclass
class RetrievalConfidence:
    """Detailed retrieval confidence metrics."""

    score: float
    avg_score: float
    score_variance: float
    agreement_ratio: float
    document_diversity: float
    metadata_consistency: float
    chunk_overlap_ratio: float
    total_results: int
    unique_documents: int
    strategy_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class AnswerConfidence:
    """Final answer confidence with all signals combined."""

    score: float
    level: str
    retrieval_confidence: float
    reranker_avg_score: float
    doc_count: int
    source_agreement: float
    signals: dict[str, float] = field(default_factory=dict)


def compute_retrieval_confidence(
    results: list[RetrievalResult],
) -> RetrievalConfidence:
    """Compute retrieval confidence from multi-strategy results.

    Uses six signals:
    1. Average similarity score
    2. Agreement between retrieval strategies (how many strategies agree)
    3. Score variance (lower variance = more consistent)
    4. Document diversity (unique vs total ratio)
    5. Metadata consistency (same filename = lower diversity penalty)
    6. Chunk overlap (consecutive chunks from same doc = good coverage)

    Args:
        results: Combined results from all retrieval strategies.

    Returns:
        Detailed retrieval confidence metrics.
    """
    if not results:
        return RetrievalConfidence(
            score=0.0, avg_score=0.0, score_variance=0.0,
            agreement_ratio=0.0, document_diversity=0.0,
            metadata_consistency=0.0, chunk_overlap_ratio=0.0,
            total_results=0, unique_documents=0, strategy_counts={},
        )

    scores = [r.score for r in results]
    avg_score = statistics.mean(scores)
    score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0

    # Normalize variance: 0 variance = 1.0, high variance approaches 0.0
    variance_signal = max(0.0, 1.0 - min(score_variance, 1.0))

    # Agreement: fraction of strategies that contributed results.
    # Deduplicated results carry merged provenance ("hybrid,similarity"),
    # so split labels to count every contributing strategy.
    strategy_counts: dict[str, int] = {}
    for r in results:
        for s in r.strategy.split(","):
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
    active_strategies = len(strategy_counts)
    total_possible = 2  # similarity, hybrid
    agreement_ratio = min(1.0, active_strategies / total_possible)

    # Document diversity: unique content hashes vs total results
    unique_contents = len({r.document.page_content for r in results})
    document_diversity = unique_contents / len(results) if results else 0.0

    # Metadata consistency: penalize if too many results from same source
    filenames = [r.document.metadata.get("filename", "") for r in results]
    unique_files = len(set(f for f in filenames if f))
    metadata_consistency = min(1.0, unique_files / max(len(results), 1))

    # Chunk overlap: consecutive chunks from same document indicate good coverage
    chunk_overlap_ratio = _compute_chunk_overlap(results)

    # Weighted combination (6 signals)
    score = (
        0.25 * min(avg_score, 1.0)
        + 0.20 * agreement_ratio
        + 0.15 * variance_signal
        + 0.15 * document_diversity
        + 0.10 * metadata_consistency
        + 0.15 * chunk_overlap_ratio
    )
    score = max(0.0, min(1.0, score))

    return RetrievalConfidence(
        score=score,
        avg_score=avg_score,
        score_variance=score_variance,
        agreement_ratio=agreement_ratio,
        document_diversity=document_diversity,
        metadata_consistency=metadata_consistency,
        chunk_overlap_ratio=chunk_overlap_ratio,
        total_results=len(results),
        unique_documents=unique_contents,
        strategy_counts=strategy_counts,
    )


def _compute_chunk_overlap(results: list[RetrievalResult]) -> float:
    """Compute chunk overlap ratio.

    Measures how many consecutive chunks from the same document appear
    in the results. This indicates good coverage of a document's content.

    Args:
        results: Retrieval results.

    Returns:
        Overlap ratio between 0 and 1.
    """
    if len(results) <= 1:
        return 0.0

    # Group by filename and check page number continuity
    by_file: dict[str, list[int]] = {}
    for r in results:
        fname = r.document.metadata.get("filename", "")
        page = r.document.metadata.get("page_number", 0)
        by_file.setdefault(fname, []).append(page)

    total_pairs = 0
    overlapping_pairs = 0

    for pages in by_file.values():
        if len(pages) < 2:
            continue
        pages_sorted = sorted(set(pages))
        for i in range(len(pages_sorted) - 1):
            total_pairs += 1
            if abs(pages_sorted[i + 1] - pages_sorted[i]) <= 1:
                overlapping_pairs += 1

    if total_pairs == 0:
        return 0.0

    return overlapping_pairs / total_pairs


def compute_answer_confidence(
    retrieval_confidence: RetrievalConfidence,
    reranker_scores: list[float] | None = None,
    llm_confidence: float | None = None,
    doc_count: int = 0,
    source_agreement: float = 0.0,
    citation_coverage: float | None = None,
    answer_completeness: float | None = None,
) -> AnswerConfidence:
    """Compute final answer confidence from all signals.

    Combines:
    1. Retrieval confidence (multi-strategy)
    2. Average reranker score (if available)
    3. LLM self-reported confidence (if available)
    4. Number of retrieved documents
    5. Source agreement (overlap between retrieval strategies)
    6. Citation coverage (fraction of sources cited in answer)
    7. Answer completeness (how thoroughly the answer addresses the question)

    Args:
        retrieval_confidence: Pre-computed retrieval confidence.
        reranker_scores: Optional list of reranker scores.
        llm_confidence: Optional LLM-reported confidence (0-1).
        doc_count: Number of documents in final context.
        source_agreement: Fraction of strategies that agreed on top results.
        citation_coverage: Fraction of retrieved sources cited in the answer (0-1).
        answer_completeness: How complete the answer is (0-1).

    Returns:
        Final answer confidence with level and detailed signals.
    """
    signals: dict[str, float] = {}

    # Signal 1: Retrieval confidence (weight: 0.35)
    signals["retrieval"] = retrieval_confidence.score

    # Signal 2: Reranker average (weight: 0.20, or 0 if unavailable)
    if reranker_scores:
        reranker_avg = statistics.mean(reranker_scores)
        signals["reranker"] = reranker_avg
    else:
        reranker_avg = 0.0
        signals["reranker"] = 0.0

    # Signal 3: LLM confidence (weight: 0.15, or 0 if unavailable)
    if llm_confidence is not None:
        signals["llm"] = llm_confidence
    else:
        signals["llm"] = 0.0

    # Signal 4: Document count signal (weight: 0.10)
    doc_signal = min(1.0, doc_count / 5.0)
    signals["doc_count"] = doc_signal

    # Signal 5: Source agreement (weight: 0.10)
    signals["source_agreement"] = source_agreement

    # Signal 6: Citation coverage (weight: 0.05)
    signals["citation_coverage"] = citation_coverage if citation_coverage is not None else 0.0

    # Signal 7: Answer completeness (weight: 0.05)
    signals["answer_completeness"] = answer_completeness if answer_completeness is not None else 0.0

    # Weighted combination: normalize by available signal weights
    # This ensures that when only a subset of signals is available,
    # the score reflects the quality of available signals (0-1 range).
    weights = {
        "retrieval": 0.35,
        "reranker": 0.20,
        "llm": 0.15,
        "doc_count": 0.10,
        "source_agreement": 0.10,
        "citation_coverage": 0.05,
        "answer_completeness": 0.05,
    }

    weighted_sum = sum(signals[k] * weights[k] for k in weights if k in signals)
    total_weight = sum(weights[k] for k in weights if k in signals)

    score = weighted_sum / total_weight if total_weight > 0 else 0.0

    score = max(0.0, min(1.0, score))

    # Map to level
    if score >= 0.7:
        level = "high"
    elif score >= 0.4:
        level = "medium"
    else:
        level = "low"

    return AnswerConfidence(
        score=score,
        level=level,
        retrieval_confidence=retrieval_confidence.score,
        reranker_avg_score=reranker_avg,
        doc_count=doc_count,
        source_agreement=source_agreement,
        signals=signals,
    )


def deduplicate_results(
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Remove duplicate documents, keeping the highest-scored version.

    Documents are deduplicated by their page_content hash.

    Args:
        results: Combined results from multiple strategies.

    Returns:
        De-duplicated results sorted by score descending.
    """
    seen: dict[str, RetrievalResult] = {}
    for r in results:
        key = r.document.page_content
        existing = seen.get(key)
        if existing is None:
            seen[key] = r
            continue
        # Same document found by another strategy: keep the best score but
        # preserve provenance as a comma-joined label ("hybrid,similarity").
        strategies = sorted(set(existing.strategy.split(",")) | set(r.strategy.split(",")))
        best = r if r.score > existing.score else existing
        seen[key] = RetrievalResult(
            document=best.document,
            score=best.score,
            strategy=",".join(strategies),
        )
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)


def normalize_scores(
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Normalize scores to 0-1 range using min-max normalization.

    Different strategies may produce scores on different scales.
    This normalizes them to a common [0, 1] range.

    Args:
        results: Results with raw scores.

    Returns:
        Results with normalized scores.
    """
    if not results:
        return results

    scores = [r.score for r in results]
    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score

    if range_score == 0:
        return [
            RetrievalResult(
                document=r.document,
                score=1.0,
                strategy=r.strategy,
            )
            for r in results
        ]

    return [
        RetrievalResult(
            document=r.document,
            score=(r.score - min_score) / range_score,
            strategy=r.strategy,
        )
        for r in results
    ]
