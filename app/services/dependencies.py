"""Dependency injection container for the application.

Provides cached singleton access to the ``RAGService`` and a reset
function for testing.
"""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings
from app.services.rag_service import RAGService


@lru_cache()
def get_rag_service() -> RAGService:
    """Return a cached singleton ``RAGService`` instance.

    The service is created on first access and reused for all subsequent
    calls within the same process.

    Returns:
        Configured ``RAGService``.
    """
    settings = get_settings()
    return RAGService(settings)


def reset_rag_service() -> None:
    """Clear the cached ``RAGService``.

    Call this in tests or when settings change at runtime.
    """
    get_rag_service.cache_clear()
