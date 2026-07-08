"""Stage 7: Context Builder.

Builds the final context for the LLM by:
- Removing redundant chunks
- Respecting token budgets
- Preferring diverse documents
- Preserving citation order
- Merging neighboring chunks from the same document
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.retriever.confidence import RetrievalResult
from app.utils.helpers import estimate_tokens
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BuiltContext:
    """Result of context building."""

    results: list[RetrievalResult]
    total_tokens: int
    chunk_count: int
    document_count: int
    merged_count: int
    dropped_count: int


class ContextBuilder:
    """Builds optimal LLM context from retrieval results."""

    def __init__(self, max_tokens: int = 3000) -> None:
        """Initialise the context builder.

        Args:
            max_tokens: Maximum total tokens for the context.
        """
        self._max_tokens = max_tokens

    def build(
        self,
        results: list[RetrievalResult],
        max_tokens: int | None = None,
        merge_neighboring: bool = True,
    ) -> BuiltContext:
        """Build the final context from retrieval results.

        Args:
            results: Ranked retrieval results.
            max_tokens: Override for token budget.
            merge_neighboring: Whether to merge adjacent chunks from same document.

        Returns:
            BuiltContext with selected results and statistics.
        """
        budget = max_tokens or self._max_tokens

        if not results:
            return BuiltContext(
                results=[],
                total_tokens=0,
                chunk_count=0,
                document_count=0,
                merged_count=0,
                dropped_count=0,
            )

        # Step 1: Remove redundant chunks (exact content duplicates)
        deduped = self._remove_redundant(results)
        dropped_redundant = len(results) - len(deduped)

        # Step 2: Merge neighboring chunks from the same document
        if merge_neighboring:
            merged, merged_count = self._merge_neighboring(deduped)
        else:
            merged = deduped
            merged_count = 0

        # Step 3: Select within token budget, preferring diversity
        selected, dropped_budget = self._select_with_budget(merged, budget)

        # Step 4: Re-sort by score (preserve citation order)
        selected.sort(key=lambda r: r.score, reverse=True)

        # Compute statistics
        total_tokens = sum(estimate_tokens(r.document.page_content) for r in selected)
        doc_files = {r.document.metadata.get("filename", "") for r in selected}

        logger.info(
            "context_built",
            selected=len(selected),
            total_tokens=total_tokens,
            budget=budget,
            dropped_redundant=dropped_redundant,
            dropped_budget=dropped_budget,
            merged=merged_count,
            documents=len(doc_files),
        )

        return BuiltContext(
            results=selected,
            total_tokens=total_tokens,
            chunk_count=len(selected),
            document_count=len(doc_files),
            merged_count=merged_count,
            dropped_count=dropped_redundant + dropped_budget,
        )

    def _remove_redundant(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Remove chunks with duplicate content, keeping highest scored.

        Args:
            results: Input results.

        Returns:
            De-duplicated results.
        """
        seen: dict[str, RetrievalResult] = {}
        for r in results:
            key = r.document.page_content
            if key not in seen or r.score > seen[key].score:
                seen[key] = r
        return list(seen.values())

    def _merge_neighboring(
        self,
        results: list[RetrievalResult],
    ) -> tuple[list[RetrievalResult], int]:
        """Merge adjacent chunks from the same document.

        If two chunks have the same filename and their page numbers are
        consecutive, merge them into a single chunk.

        Args:
            results: De-duplicated results.

        Returns:
            Tuple of (merged results, count of merges).
        """
        if len(results) <= 1:
            return results, 0

        # Group by filename
        by_file: dict[str, list[RetrievalResult]] = {}
        for r in results:
            fname = r.document.metadata.get("filename", "")
            by_file.setdefault(fname, []).append(r)

        merged_results: list[RetrievalResult] = []
        merge_count = 0

        for fname, file_results in by_file.items():
            # Sort by page number
            file_results.sort(
                key=lambda r: r.document.metadata.get("page_number", 0),
            )

            current_group: list[RetrievalResult] = [file_results[0]]

            for i in range(1, len(file_results)):
                prev_page = current_group[-1].document.metadata.get("page_number", 0)
                curr_page = file_results[i].document.metadata.get("page_number", 0)

                # Merge if same page or adjacent pages
                if abs(curr_page - prev_page) <= 1:
                    current_group.append(file_results[i])
                else:
                    merged_results.append(self._merge_group(current_group))
                    merge_count += len(current_group) - 1
                    current_group = [file_results[i]]

            merged_results.append(self._merge_group(current_group))
            merge_count += len(current_group) - 1

        return merged_results, merge_count

    def _merge_group(
        self,
        group: list[RetrievalResult],
    ) -> RetrievalResult:
        """Merge a group of results into a single result.

        Args:
            group: Group of results to merge.

        Returns:
            Single merged result.
        """
        if len(group) == 1:
            return group[0]

        # Merge content
        contents = [r.document.page_content for r in group]
        merged_content = "\n\n".join(contents)

        # Use highest score
        best = max(group, key=lambda r: r.score)

        # Merge metadata
        merged_metadata = dict(best.document.metadata)
        pages = [r.document.metadata.get("page_number", 0) for r in group]
        merged_metadata["page_number"] = min(pages)
        merged_metadata["merged_pages"] = pages
        merged_metadata["merged_count"] = len(group)

        from langchain_core.documents import Document

        merged_doc = Document(
            page_content=merged_content,
            metadata=merged_metadata,
        )

        return RetrievalResult(
            document=merged_doc,
            score=best.score,
            strategy=best.strategy,
        )

    def _select_with_budget(
        self,
        results: list[RetrievalResult],
        budget: int,
    ) -> tuple[list[RetrievalResult], int]:
        """Select results within token budget, preferring document diversity.

        Uses a greedy approach: alternate between different documents
        to maximize diversity while respecting the token limit.

        Args:
            results: Results to select from.
            budget: Maximum token budget.

        Returns:
            Tuple of (selected results, count dropped due to budget).
        """
        if not results:
            return [], 0

        # Round-robin by document for diversity
        by_file: dict[str, list[RetrievalResult]] = {}
        for r in results:
            fname = r.document.metadata.get("filename", "unknown")
            by_file.setdefault(fname, [])

        # Sort results within each file by score
        for fname in by_file:
            by_file[fname] = sorted(
                [r for r in results if r.document.metadata.get("filename", "") == fname],
                key=lambda r: r.score,
                reverse=True,
            )

        selected: list[RetrievalResult] = []
        total_tokens = 0
        file_iters = {k: iter(v) for k, v in by_file.items()}
        remaining = {k: len(v) for k, v in by_file.items()}

        while any(remaining.values()):
            for fname in list(remaining.keys()):
                if remaining[fname] <= 0:
                    continue
                try:
                    result = next(file_iters[fname])
                    chunk_tokens = estimate_tokens(result.document.page_content)
                    if total_tokens + chunk_tokens <= budget:
                        selected.append(result)
                        total_tokens += chunk_tokens
                    remaining[fname] -= 1
                except StopIteration:
                    remaining[fname] = 0

        dropped = len(results) - len(selected)
        return selected, dropped
