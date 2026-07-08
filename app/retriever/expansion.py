"""Stage 5: Query Expansion.

Expands low-confidence queries using multiple strategies:
- Multi-Query: LLM-generated alternative phrasings
- HyDE: Hypothetical Document Embeddings
- Synonym Expansion: WordNet-based synonym augmentation
- Keyword Expansion: Extract and expand key terms
"""

from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.config.constants import HYDE_PROMPT, MULTI_QUERY_PROMPT
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Simple synonym map for common terms (no NLTK dependency)
_SYNONYM_MAP: dict[str, list[str]] = {
    "fast": ["quick", "rapid", "speedy"],
    "slow": ["gradual", "sluggish"],
    "big": ["large", "huge", "massive"],
    "small": ["tiny", "little", "minute"],
    "good": ["excellent", "great", "quality"],
    "bad": ["poor", "terrible", "defective"],
    "important": ["critical", "essential", "crucial"],
    "help": ["assist", "support", "aid"],
    "show": ["display", "present", "demonstrate"],
    "find": ["search", "locate", "discover"],
    "create": ["build", "generate", "construct"],
    "change": ["modify", "update", "alter"],
    "problem": ["issue", "error", "bug"],
    "use": ["utilize", "employ", "apply"],
    "make": ["produce", "generate", "build"],
    "get": ["retrieve", "obtain", "fetch"],
    "set": ["configure", "establish", "define"],
    "run": ["execute", "perform", "operate"],
    "check": ["verify", "validate", "inspect"],
    "data": ["information", "records", "dataset"],
    "system": ["platform", "framework", "architecture"],
    "model": ["algorithm", "network", "architecture"],
    "train": ["educate", "teach", "fit"],
    "test": ["evaluate", "assess", "validate"],
    "result": ["outcome", "output", "finding"],
    "method": ["approach", "technique", "procedure"],
    "process": ["workflow", "procedure", "pipeline"],
    "function": ["method", "routine", "procedure"],
    "variable": ["parameter", "attribute", "field"],
    "error": ["bug", "issue", "fault"],
    "feature": ["capability", "functionality", "attribute"],
}


class QueryExpander:
    """Expands queries using multiple strategies."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """Initialise the expander.

        Args:
            llm: LLM for multi-query and HyDE generation.
        """
        self._llm = llm

    def expand_multi_query(
        self,
        query: str,
        num_queries: int = 3,
    ) -> list[str]:
        """Generate alternative query phrasings using the LLM.

        Args:
            query: Original query.
            num_queries: Number of alternatives to generate.

        Returns:
            List of expanded query strings (includes original).
        """
        if self._llm is None:
            return [query]

        prompt = MULTI_QUERY_PROMPT.format(
            num_queries=num_queries,
            question=query,
        )
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            queries = [
                line.split(".", 1)[-1].strip()
                for line in text.strip().split("\n")
                if line.strip() and "." in line
            ]
            result = queries if queries else [query]
            logger.info(
                "multi_query_expansion",
                original=query[:50],
                expanded_count=len(result),
            )
            return result
        except Exception as exc:
            logger.warning("multi_query_failed", error=str(exc))
            return [query]

    def expand_hyde(self, query: str) -> str:
        """Generate a hypothetical answer for HyDE retrieval.

        Args:
            query: Original query.

        Returns:
            Hypothetical answer text.
        """
        if self._llm is None:
            return query

        prompt = HYDE_PROMPT.format(question=query)
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            hypothetical = response.content if hasattr(response, "content") else str(response)
            logger.info(
                "hyde_generated",
                original_length=len(query),
                hyde_length=len(hypothetical),
            )
            return hypothetical
        except Exception as exc:
            logger.warning("hyde_failed", error=str(exc))
            return query

    def expand_synonyms(self, query: str, max_synonyms: int = 3) -> str:
        """Expand query with synonyms for key terms.

        Args:
            query: Original query.
            max_synonyms: Maximum number of synonym expansions.

        Returns:
            Expanded query with synonyms.
        """
        words = query.split()
        expanded_words = list(words)
        expansions = 0

        for i, word in enumerate(words):
            lower = word.lower().strip("?.!,;:'\"")
            if lower in _SYNONYM_MAP and expansions < max_synonyms:
                synonyms = _SYNONYM_MAP[lower]
                expanded_words.insert(i + 1, synonyms[0])
                expansions += 1

        result = " ".join(expanded_words)
        if expansions > 0:
            logger.info("synonym_expansion", expansions=expansions)
        return result

    def expand_keywords(self, query: str) -> list[str]:
        """Extract and expand key terms from the query.

        Identifies important nouns/phrases and generates variations.

        Args:
            query: Original query.

        Returns:
            List of keyword variations.
        """
        # Simple keyword extraction: remove stop words, keep significant terms
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own",
            "same", "than", "too", "very", "just", "because", "about",
        }

        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Generate keyword variations
        variations = list(keywords)
        for kw in keywords[:5]:
            if kw in _SYNONYM_MAP:
                variations.extend(_SYNONYM_MAP[kw][:2])

        return list(dict.fromkeys(variations))  # dedupe preserving order

    def expand_all(
        self,
        query: str,
        num_multi_query: int = 3,
    ) -> dict[str, Any]:
        """Run all expansion strategies and return combined results.

        Args:
            query: Original query.
            num_multi_query: Number of multi-query alternatives.

        Returns:
            Dict with 'multi_query', 'hyde', 'synonyms', 'keywords'.
        """
        return {
            "multi_query": self.expand_multi_query(query, num_multi_query),
            "hyde": self.expand_hyde(query),
            "synonyms": self.expand_synonyms(query),
            "keywords": self.expand_keywords(query),
        }
