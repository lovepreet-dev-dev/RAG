"""Stage 1: Query Classification.

Classifies user queries into types to route to optimal retrieval strategies.
Uses rule-based heuristics with optional LLM classification for ambiguous cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class QueryType(str, Enum):
    """Classification of user query types."""

    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    REASONING = "reasoning"
    CODE = "code"
    MATH = "math"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedQuery:
    """Result of query classification."""

    query_type: QueryType
    confidence: float
    suggested_strategies: list[str]
    suggested_top_k: int


# Strategy mappings per query type
_STRATEGY_MAP: dict[QueryType, list[str]] = {
    QueryType.FACTUAL: ["similarity", "hybrid"],
    QueryType.CONVERSATIONAL: ["similarity"],
    QueryType.SUMMARIZATION: ["similarity", "hybrid"],
    QueryType.COMPARISON: ["similarity", "hybrid"],
    QueryType.REASONING: ["similarity", "hybrid"],
    QueryType.CODE: ["similarity", "hybrid"],
    QueryType.MATH: ["similarity"],
}

_TOP_K_MAP: dict[QueryType, int] = {
    QueryType.FACTUAL: 5,
    QueryType.CONVERSATIONAL: 3,
    QueryType.SUMMARIZATION: 10,
    QueryType.COMPARISON: 8,
    QueryType.REASONING: 7,
    QueryType.CODE: 5,
    QueryType.MATH: 5,
}


class QueryClassifier:
    """Classifies queries using rule-based heuristics."""

    # Weighted keyword patterns per type.  Intent verbs (compare, summarize,
    # why, …) weigh more than topic keywords (python, sql, …) so that
    # "Compare Python and Java" classifies as COMPARISON, not CODE.
    _PATTERNS: dict[QueryType, list[tuple[re.Pattern[str], int]]] = {
        QueryType.FACTUAL: [
            (re.compile(r"\b(what|who|when|where|which|is|are|was|were)\b", re.I), 2),
            (re.compile(r"\b(define|definition|meaning|explain)\b", re.I), 3),
        ],
        QueryType.CONVERSATIONAL: [
            (re.compile(r"\b(hi|hello|hey|thanks|thank you|bye|goodbye)\b", re.I), 3),
            (re.compile(r"\b(how are you|what's up|help me)\b", re.I), 3),
        ],
        QueryType.SUMMARIZATION: [
            (re.compile(r"\b(summarize|summary|overview|tldr|tl;dr|brief)\b", re.I), 3),
            (re.compile(r"\b(give me (a )?summary|key points|main ideas)\b", re.I), 3),
        ],
        QueryType.COMPARISON: [
            (re.compile(r"\b(compare|versus|vs\.?|difference|differ|contrast)\b", re.I), 3),
            (re.compile(r"\b(better|worse|pros and cons|advantages)\b", re.I), 2),
        ],
        QueryType.REASONING: [
            (re.compile(r"\b(why|how does|explain why|reason|because)\b", re.I), 3),
            (re.compile(r"\b(cause|effect|implication|consequence)\b", re.I), 2),
        ],
        QueryType.CODE: [
            (re.compile(r"\b(code|function|class|implement|debug|error|bug)\b", re.I), 3),
            (re.compile(r"\b(python|javascript|java|sql|html|css|api)\b", re.I), 1),
            (re.compile(r"```|\bdef\b|\bimport\b", re.I), 3),
        ],
        QueryType.MATH: [
            (re.compile(r"\b(calculate|compute|solve|equation|formula)\b", re.I), 3),
            (re.compile(r"\b(sum|average|mean|median|percent|ratio)\b", re.I), 2),
            (re.compile(r"[\d]+\s*[\+\-\*/\^=]\s*[\d]+", re.I), 3),
        ],
    }

    def classify(self, query: str) -> ClassifiedQuery:
        """Classify a query into its type.

        Args:
            query: Preprocessed query string.

        Returns:
            ClassifiedQuery with type, confidence, and suggestions.
        """
        if not query or not query.strip():
            return ClassifiedQuery(
                query_type=QueryType.CONVERSATIONAL,
                confidence=1.0,
                suggested_strategies=["similarity"],
                suggested_top_k=3,
            )

        scores: dict[QueryType, int] = {}
        for qtype, patterns in self._PATTERNS.items():
            score = 0
            for pattern, weight in patterns:
                score += weight * len(pattern.findall(query))
            scores[qtype] = score

        # Find best match
        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_type]

        if best_score == 0:
            best_type = QueryType.FACTUAL
            confidence = 0.5
        else:
            total = sum(scores.values())
            confidence = min(1.0, best_score / max(total, 1))

        return ClassifiedQuery(
            query_type=best_type,
            confidence=confidence,
            suggested_strategies=_STRATEGY_MAP[best_type],
            suggested_top_k=_TOP_K_MAP[best_type],
        )
