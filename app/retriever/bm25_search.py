"""BM25 lexical search for hybrid retrieval.

Provides a lightweight BM25 implementation that builds an inverted index
from stored documents.  Combined with dense vector search via Reciprocal
Rank Fusion (RRF) for true hybrid retrieval.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from langchain_core.documents import Document


class BM25Search:
    """Lightweight BM25 search over a document collection.

    Builds an inverted index from documents and scores them using the
    BM25 ranking function (Okapi BM25 variant).

    Args:
        k1: Term frequency saturation parameter (default 1.5).
        b: Length normalization parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._documents: list[Document] = []
        self._doc_count: int = 0
        self._idf_cache: dict[str, float] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return re.findall(r"\w+", text.lower())

    def build_index(self, documents: list[Document]) -> None:
        """Build the BM25 inverted index from documents.

        Args:
            documents: List of LangChain Document objects to index.
        """
        self._documents = documents
        self._doc_count = len(documents)
        self._inverted_index.clear()
        self._doc_lengths.clear()
        self._idf_cache.clear()

        total_length = 0
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc.page_content)
            self._doc_lengths.append(len(tokens))
            total_length += len(tokens)

            # Build inverted index: term -> [(doc_idx, term_freq)]
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self._inverted_index[term].append((doc_idx, count))

        self._avg_doc_length = total_length / max(self._doc_count, 1)

    def _idf(self, term: str) -> float:
        """Compute inverse document frequency for a term."""
        if term in self._idf_cache:
            return self._idf_cache[term]

        df = len(self._inverted_index.get(term, []))
        # IDF with smoothing
        idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)
        self._idf_cache[term] = idf
        return idf

    def search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """Search the index with a BM25 query.

        Args:
            query: Search query string.
            k: Number of top results to return.

        Returns:
            List of (Document, score) tuples sorted by BM25 score descending.
        """
        if not self._documents or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self._inverted_index:
                continue

            idf = self._idf(term)
            posting_list = self._inverted_index[term]

            for doc_idx, term_freq in posting_list:
                doc_len = self._doc_lengths[doc_idx]
                # BM25 scoring formula
                numerator = term_freq * (self._k1 + 1)
                denominator = term_freq + self._k1 * (
                    1 - self._b + self._b * doc_len / max(self._avg_doc_length, 1)
                )
                scores[doc_idx] += idf * numerator / denominator

        # Sort by score descending and return top k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self._documents[idx], score) for idx, score in ranked]

    @property
    def doc_count(self) -> int:
        """Return the number of indexed documents."""
        return self._doc_count


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF scores each document as: sum(1 / (k + rank_i)) across all lists
    where the document appears.  This is effective for combining heterogeneous
    ranking signals (e.g., BM25 lexical + dense vector).

    Reference: Cormack et al., "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods" (SIGIR 2009).

    Args:
        ranked_lists: List of ranked result lists, each containing
            (Document, score) tuples.
        k: RRF constant (default 60, as recommended in the paper).

    Returns:
        Fused list of (Document, float) sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked):
            doc_key = doc.page_content
            rrf_scores[doc_key] += 1.0 / (k + rank + 1)
            doc_map[doc_key] = doc

    ranked_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in ranked_fused]


def weighted_score_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    weights: list[float] | None = None,
) -> list[tuple[Document, float]]:
    """Combine ranked lists using weighted linear combination.

    Normalizes scores within each list to [0, 1] before weighting.

    Args:
        ranked_lists: List of ranked result lists.
        weights: Optional weights for each list. Defaults to equal weights.

    Returns:
        Fused list sorted by combined score descending.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0 / len(ranked_lists)] * len(ranked_lists)
    else:
        total = sum(weights)
        weights = [w / total for w in weights] if total > 0 else [1.0 / len(ranked_lists)] * len(ranked_lists)

    combined_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked, weight in zip(ranked_lists, weights):
        if not ranked:
            continue
        scores = [s for _, s in ranked]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s

        for doc, score in ranked:
            normalized = (score - min_s) / range_s if range_s > 0 else 1.0
            combined_scores[doc.page_content] += weight * normalized
            doc_map[doc.page_content] = doc

    ranked_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in ranked_combined]
