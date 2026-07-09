"""Context compression for RAG pipeline.

Compresses retrieved context before sending to the LLM to reduce token
usage while preserving key information, citations, and important facts.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.retriever.confidence import RetrievalResult
from app.utils.helpers import estimate_tokens
from app.utils.logging import get_logger

logger = get_logger(__name__)

_COMPRESSION_SYSTEM_PROMPT = """You are a context compression assistant. Your job is to
compress the following retrieved context passages while preserving ALL of the following:
1. Key facts, numbers, dates, and specific claims
2. Source citations in the format [Source N: filename, Page X]
3. Technical terms and proper nouns
4. Any information directly relevant to the question

Remove:
- Redundant or duplicate information across passages
- Filler words and unnecessary elaboration
- Passages that are clearly irrelevant to the question

Output ONLY the compressed context, preserving the [Source N] citation format.
Do not add any explanation or commentary."""


def compress_context(
    results: list[RetrievalResult],
    question: str,
    llm: BaseChatModel | None = None,
    max_tokens: int = 2000,
) -> str:
    """Compress retrieved context while preserving key information.

    If no LLM is provided or compression is disabled, returns a simple
    concatenation of the context.

    Args:
        results: Ranked retrieval results.
        question: The user's question (for relevance filtering).
        llm: Optional LLM for compression.
        max_tokens: Target maximum tokens for compressed output.

    Returns:
        Compressed context string.
    """
    if not results:
        return ""

    # Build raw context
    raw_parts: list[str] = []
    for i, r in enumerate(results):
        source_label = (
            f"[Source {i+1}: {r.document.metadata.get('filename', 'unknown')}, "
            f"Page {r.document.metadata.get('page_number', '?')}]"
        )
        raw_parts.append(f"{source_label}\n{r.document.page_content}")

    raw_context = "\n\n".join(raw_parts)
    raw_tokens = estimate_tokens(raw_context)

    # If already within budget or no LLM available, return as-is
    if llm is None or raw_tokens <= max_tokens:
        return raw_context

    # Compress using LLM
    try:
        human_content = (
            f"Original context ({raw_tokens} tokens):\n\n{raw_context}\n\n"
            f"Question: {question}\n\n"
            f"Compress this to under {max_tokens} tokens while preserving "
            "key facts and citations."
        )

        response = llm.invoke([
            SystemMessage(content=_COMPRESSION_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ])

        compressed = response.content if hasattr(response, "content") else str(response)
        compressed_tokens = estimate_tokens(compressed)

        logger.info(
            "context_compressed",
            original_tokens=raw_tokens,
            compressed_tokens=compressed_tokens,
            reduction_pct=round((1 - compressed_tokens / raw_tokens) * 100, 1),
        )

        return compressed.strip()

    except Exception as exc:
        logger.warning("context_compression_failed", error=str(exc))
        return raw_context
