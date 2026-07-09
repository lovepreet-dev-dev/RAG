"""Stage 9: Answer Validation.

Validates LLM-generated answers by:
- Verifying citations match source documents
- Detecting unsupported statements
- Calibrating confidence based on answer quality
- Checking source coverage
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.retriever.confidence import RetrievalResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of answer validation."""

    is_valid: bool
    citation_score: float
    coverage_score: float
    hallucination_score: float
    overall_score: float
    issues: list[str] = field(default_factory=list)
    unsupported_statements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "citation_score": round(self.citation_score, 4),
            "coverage_score": round(self.coverage_score, 4),
            "hallucination_score": round(self.hallucination_score, 4),
            "overall_score": round(self.overall_score, 4),
            "issues": self.issues,
            "unsupported_statements": self.unsupported_statements,
        }


class AnswerValidator:
    """Validates LLM answers against source documents."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """Initialise the validator.

        Args:
            llm: LLM for hallucination detection (optional).
        """
        self._llm = llm

    def validate(
        self,
        answer: str,
        sources: list[RetrievalResult],
        query: str,
    ) -> ValidationResult:
        """Validate an answer against its sources.

        Args:
            answer: LLM-generated answer.
            sources: Source documents used for generation.
            query: Original user query.

        Returns:
            ValidationResult with scores and issues.
        """
        if not answer:
            return ValidationResult(
                is_valid=False,
                citation_score=0.0,
                coverage_score=0.0,
                hallucination_score=0.0,
                overall_score=0.0,
                issues=["Empty answer"],
            )

        # Check 1: Citation verification
        citation_score, citation_issues = self._verify_citations(answer, sources)

        # Check 2: Source coverage
        coverage_score = self._check_coverage(answer, sources)

        # Check 3: Hallucination detection
        hallucination_score, unsupported = self._detect_hallucinations(
            answer, sources, query,
        )

        # Overall score
        overall = (
            0.4 * citation_score
            + 0.3 * coverage_score
            + 0.3 * hallucination_score
        )

        issues = citation_issues
        if unsupported:
            issues.append(f"Potentially unsupported statements: {len(unsupported)}")

        is_valid = overall >= 0.5 and len(unsupported) == 0

        logger.info(
            "answer_validated",
            is_valid=is_valid,
            citation_score=round(citation_score, 3),
            coverage_score=round(coverage_score, 3),
            hallucination_score=round(hallucination_score, 3),
            overall=round(overall, 3),
            issues_count=len(issues),
        )

        return ValidationResult(
            is_valid=is_valid,
            citation_score=citation_score,
            coverage_score=coverage_score,
            hallucination_score=hallucination_score,
            overall_score=overall,
            issues=issues,
            unsupported_statements=unsupported,
        )

    def _verify_citations(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> tuple[float, list[str]]:
        """Verify that citations in the answer match source documents.

        Checks for patterns like:
        - [Source 1: filename, Page X]
        - (filename, page X)
        - "from the document"

        Args:
            answer: LLM-generated answer.
            sources: Source documents.

        Returns:
            Tuple of (citation score, list of issues).
        """
        issues: list[str] = []

        # Extract referenced filenames from answer
        cited_files = set()

        # Pattern: [Source N: filename, Page X]
        source_pattern = re.findall(
            r"\[Source\s+\d+:\s*([^,\]]+)", answer, re.I,
        )
        cited_files.update(f.strip() for f in source_pattern)

        # Pattern: "filename" or 'filename' mentioned
        quoted = re.findall(r'["\']([^"\']+\.\w+)["\']', answer)
        cited_files.update(quoted)

        # Get actual source filenames
        actual_files = {
            r.document.metadata.get("filename", "") for r in sources
        }

        # Score: fraction of cited files that are real sources
        if cited_files:
            valid_citations = cited_files & actual_files
            score = len(valid_citations) / len(cited_files)
            invalid = cited_files - actual_files
            if invalid:
                issues.append(f"Invalid citations: {invalid}")
        else:
            # No citations found — mild penalty
            score = 0.7
            issues.append("No explicit citations found in answer")

        return score, issues

    def _check_coverage(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> float:
        """Check how well the answer covers the source material.

        Uses keyword overlap between answer and source content.

        Args:
            answer: LLM-generated answer.
            sources: Source documents.

        Returns:
            Coverage score between 0 and 1.
        """
        if not sources:
            return 0.5

        answer_words = set(answer.lower().split())
        source_words = set()
        for r in sources:
            source_words.update(r.document.page_content.lower().split())

        if not source_words:
            return 0.5

        # How much of the source material is referenced in the answer
        overlap = len(answer_words & source_words)
        coverage = min(1.0, overlap / max(len(source_words) * 0.3, 1))

        return coverage

    def _detect_hallucinations(
        self,
        answer: str,
        sources: list[RetrievalResult],
        query: str,
    ) -> tuple[float, list[str]]:
        """Detect potentially hallucinated statements.

        Uses the LLM to check if statements are supported by sources,
        or falls back to keyword-based detection.

        Args:
            answer: LLM-generated answer.
            sources: Source documents.
            query: Original query.

        Returns:
            Tuple of (hallucination score, list of unsupported statements).
        """
        if self._llm is not None:
            return self._llm_hallucination_check(answer, sources, query)
        return self._keyword_hallucination_check(answer, sources)

    def _llm_hallucination_check(
        self,
        answer: str,
        sources: list[RetrievalResult],
        query: str,
    ) -> tuple[float, list[str]]:
        """Use LLM to detect hallucinations.

        Args:
            answer: LLM-generated answer.
            sources: Source documents.
            query: Original query.

        Returns:
            Tuple of (score, unsupported statements).
        """
        context = "\n\n".join(
            f"[Source {i+1}: {r.document.metadata.get('filename', 'unknown')}]\n"
            f"{r.document.page_content}"
            for i, r in enumerate(sources[:5])
        )

        prompt = f"""You are a strict fact-checking assistant. Given a question, answer, and source documents,
determine which statements in the answer are NOT supported by the sources.

Question: {query}

Answer to check:
{answer}

Source documents:
{context}

Rules:
1. A statement is SUPPORTED if the information can be reasonably inferred from the sources.
2. A statement is UNSUPPORTED if it contains specific facts, numbers, dates, or claims not present in the sources.
3. A statement is UNSUPPORTED if it fabricates citations or attributes information to the wrong source.
4. If the answer says "I don't know" or "Information not found", that is VALID — not a hallucination.
5. Be strict — only mark as unsupported if the information genuinely cannot be found or inferred from the sources.

List each unsupported statement on a new line. If all statements are supported, reply with "NONE".
Format: "UNSUPPORTED: <statement>" for each unsupported statement."""

        try:
            response = self._llm.invoke([
                SystemMessage(content="You are a fact-checking assistant."),
                HumanMessage(content=prompt),
            ])
            text = response.content if hasattr(response, "content") else str(response)

            if "NONE" in text.upper():
                return 1.0, []

            unsupported = [
                line.strip().lstrip("0123456789.-) ")
                for line in text.strip().split("\n")
                if line.strip() and "NONE" not in line.upper()
            ]
            score = max(0.0, 1.0 - len(unsupported) * 0.2)
            return score, unsupported

        except Exception as exc:
            logger.warning("llm_hallucination_check_failed", error=str(exc))
            return self._keyword_hallucination_check(answer, sources)

    def _keyword_hallucination_check(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> tuple[float, list[str]]:
        """Keyword-based hallucination detection (fallback).

        Flags sentences that contain:
        1. Numbers/statistics not found in sources
        2. Key nouns/phrases not found in sources

        Args:
            answer: LLM-generated answer.
            sources: Source documents.

        Returns:
            Tuple of (score, unsupported statements).
        """
        source_text = " ".join(r.document.page_content for r in sources)
        source_words = set(source_text.lower().split())

        # Find numeric claims in answer
        answer_numbers = set(re.findall(r"\b\d+\.?\d*%?\b", answer))
        source_numbers = set(re.findall(r"\b\d+\.?\d*%?\b", source_text))

        unsupported_numbers = answer_numbers - source_numbers

        sentences = answer.split(".")
        unsupported = []
        for sent in sentences:
            sent_stripped = sent.strip()
            if len(sent_stripped) < 10:
                continue

            # Check 1: Numeric claims not in source
            for num in unsupported_numbers:
                if num in sent:
                    unsupported.append(sent_stripped)
                    break
            else:
                # Check 2: Key content words not in source
                sent_words = set(re.findall(r"\b[a-z]{4,}\b", sent_stripped.lower()))
                # Remove common stop words
                stop_words = {"this", "that", "with", "from", "have", "been", "were", "they", "their", "about", "which", "would", "could", "should", "there", "also", "than", "then", "some", "into", "only", "very", "most", "when", "what", "your", "will", "each", "make", "like", "over", "such"}
                content_words = sent_words - stop_words

                if content_words:
                    # If more than half of content words are not in source, flag it
                    missing = content_words - source_words
                    if len(missing) > len(content_words) * 0.5 and len(content_words) >= 2:
                        unsupported.append(sent_stripped)

        score = max(0.0, 1.0 - len(unsupported) * 0.15)
        return score, unsupported
