"""Utility helpers for file handling, hashing, text processing, and time tracking.

Every function is pure (no side effects beyond explicitly documented ones)
and has full type hints.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from app.config.constants import SUPPORTED_EXTENSIONS


def generate_id() -> str:
    """Generate a unique identifier string (UUID4).

    Returns:
        A lowercase UUID4 string.
    """
    return str(uuid.uuid4())


def generate_content_hash(content: str | bytes) -> str:
    """Generate a SHA-256 hash of content for duplicate detection.

    Args:
        content: The string or bytes content to hash.

    Returns:
        64-character hex digest of the SHA-256 hash.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def clean_text(text: str) -> str:
    """Clean and normalise extracted text.

    * Normalises ``\\r\\n`` to ``\\n``.
    * Collapses runs of non-newline whitespace to a single space.
    * Reduces three or more consecutive newlines to two.
    * Strips leading/trailing whitespace.

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned text.
    """
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def get_file_extension(filename: str) -> str:
    """Return the lowercase file extension including the dot.

    Args:
        filename: The file name or path.

    Returns:
        Lowercase extension with dot, e.g. ``'.pdf'``.
    """
    return Path(filename).suffix.lower()


def is_supported_file(filename: str) -> bool:
    """Check whether a file extension is supported for ingestion.

    Args:
        filename: The file name.

    Returns:
        ``True`` if the file type is supported.
    """
    ext = get_file_extension(filename)
    return ext in SUPPORTED_EXTENSIONS


def get_file_type(filename: str) -> Optional[str]:
    """Map a filename to its internal type identifier.

    Args:
        filename: The file name.

    Returns:
        Type string (e.g. ``'pdf'``, ``'text'``) or ``None`` if unsupported.
    """
    ext = get_file_extension(filename)
    return SUPPORTED_EXTENSIONS.get(ext)


def safe_filename(filename: str) -> str:
    """Sanitise a filename by removing path separators and null bytes.

    Args:
        filename: The original filename.

    Returns:
        Sanitised filename safe for filesystem use.
    """
    name = Path(filename).name
    name = name.replace("\x00", "")
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "unnamed_file"


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length, appending ellipsis if needed.

    Truncation happens at the nearest word boundary to avoid mid-word cuts.

    Args:
        text: The text to truncate.
        max_length: Maximum character count.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rsplit(" ", 1)[0] + "..."


def extract_page_number(text: str, fallback: int = 0) -> int:
    """Attempt to extract a page number from text or metadata.

    Looks for patterns like ``page 5``, ``p. 12``, ``Page 3``.

    Args:
        text: Text that may contain page markers.
        fallback: Default page number if none found.

    Returns:
        Extracted page number or the fallback.
    """
    match = re.search(r"(?:page|p\.?)\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return fallback


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (approximately 1 token per 4 characters).

    Args:
        text: The text to estimate.

    Returns:
        Approximate token count (minimum 1).
    """
    return max(1, len(text) // 4)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    cost_per_1k_input: float = 0.005,
    cost_per_1k_output: float = 0.015,
) -> float:
    """Estimate the cost in USD for a given token split.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cost_per_1k_input: Cost per 1k input tokens.
        cost_per_1k_output: Cost per 1k output tokens.

    Returns:
        Estimated cost in USD.
    """
    input_cost = (input_tokens / 1000) * cost_per_1k_input
    output_cost = (output_tokens / 1000) * cost_per_1k_output
    return round(input_cost + output_cost, 8)


class Timer:
    """Context-manager timer for measuring block execution time.

    Usage::

        with Timer() as t:
            do_work()
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self._end = time.perf_counter()
        self.elapsed_ms = round((self._end - self._start) * 1000, 2)
