"""Stage 0: Query Preprocessing.

Validates, normalizes, and enriches user queries before retrieval.
Extracts metadata filters, detects language, and integrates conversation context.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PreprocessedQuery:
    """Result of query preprocessing."""

    original: str
    normalized: str
    language: str
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    validation_error: str = ""
    tokens_estimate: int = 0


class QueryPreprocessor:
    """Preprocesses user queries for optimal retrieval."""

    # Common noise patterns to strip
    _NOISE_PATTERNS = [
        (r"\s+", " "),                    # collapse whitespace
        (r"[^\w\s?.!,;:'\"-]", ""),      # remove special chars except basics
    ]

    # Metadata filter patterns: "file:report.pdf", "page:3", "source:docs"
    _METADATA_PATTERNS = {
        "filename": r"file:(\S+)",
        "page_number": r"page:(\d+)",
        "source": r"source:(\S+)",
        "document_id": r"doc:(\S+)",
    }

    # Language detection heuristics (simple keyword-based)
    _LANG_KEYWORDS: dict[str, list[str]] = {
        "en": ["the", "is", "are", "was", "what", "how", "why", "when", "where", "which"],
        "es": ["el", "la", "los", "las", "es", "son", "qué", "cómo", "por qué"],
        "fr": ["le", "la", "les", "est", "sont", "quoi", "comment", "pourquoi"],
        "de": ["der", "die", "das", "ist", "sind", "was", "wie", "warum"],
    }

    def preprocess(
        self,
        query: str,
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> PreprocessedQuery:
        """Full preprocessing pipeline.

        Args:
            query: Raw user query.
            chat_history: Optional conversation history.

        Returns:
            PreprocessedQuery with normalized text and metadata.
        """
        if not query or not query.strip():
            return PreprocessedQuery(
                original=query or "",
                normalized="",
                language="en",
                is_valid=False,
                validation_error="Empty query",
            )

        # Step 1: Normalize
        normalized = self._normalize(query)

        # Step 2: Extract metadata filters
        metadata_filters, cleaned = self._extract_metadata(normalized)

        # Step 3: Detect language
        language = self._detect_language(cleaned)

        # Step 4: Integrate conversation context if query is very short
        if chat_history and len(cleaned.split()) < 3:
            cleaned = self._enrich_with_context(cleaned, chat_history)

        # Step 5: Estimate tokens (rough: 1 token ≈ 4 chars)
        tokens_estimate = max(1, len(cleaned) // 4)

        return PreprocessedQuery(
            original=query,
            normalized=cleaned,
            language=language,
            metadata_filters=metadata_filters,
            is_valid=True,
            tokens_estimate=tokens_estimate,
        )

    def _normalize(self, query: str) -> str:
        """Normalize query text.

        - Unicode normalization
        - Strip whitespace
        - Collapse multiple spaces
        - Remove leading/trailing punctuation

        Args:
            query: Raw query.

        Returns:
            Normalized query.
        """
        # Unicode normalize
        text = unicodedata.normalize("NFKC", query)

        # Apply noise patterns
        for pattern, replacement in self._NOISE_PATTERNS:
            text = re.sub(pattern, replacement, text)

        # Strip and clean edges
        text = text.strip()
        text = text.strip("?.!,;:'\"-")

        return text

    def _extract_metadata(self, query: str) -> tuple[dict[str, Any], str]:
        """Extract metadata filter expressions from the query.

        Recognizes patterns like:
        - file:report.pdf
        - page:3
        - source:docs
        - doc:abc123

        Args:
            query: Normalized query.

        Returns:
            Tuple of (metadata_filters dict, cleaned query).
        """
        filters: dict[str, Any] = {}
        cleaned = query

        for key, pattern in self._METADATA_PATTERNS.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if key == "page_number":
                    filters[key] = int(matches[0])
                else:
                    filters[key] = matches[0]
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        return filters, cleaned

    def _detect_language(self, query: str) -> str:
        """Detect query language using keyword heuristics.

        Args:
            query: Cleaned query.

        Returns:
            ISO 639-1 language code (default: 'en').
        """
        words = set(query.lower().split())
        scores: dict[str, int] = {}

        for lang, keywords in self._LANG_KEYWORDS.items():
            scores[lang] = len(words & set(keywords))

        if not scores or max(scores.values()) == 0:
            return "en"

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _enrich_with_context(
        self,
        query: str,
        chat_history: list[dict[str, str]],
    ) -> str:
        """Enrich a short query with conversation context.

        If the query is very short (e.g. "it", "that", "more"), prepend
        the last assistant message for context.

        Args:
            query: Short query.
            chat_history: Conversation history.

        Returns:
            Enriched query.
        """
        if not chat_history:
            return query

        # Get last assistant message
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                context = msg["content"][:200]
                return f"Given the context about '{context}...', answer: {query}"

        return query
