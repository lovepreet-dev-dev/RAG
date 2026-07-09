"""Background job functions for document ingestion and maintenance."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


def ingest_documents_background(
    file_paths: list[str],
    chunking_strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict:
    """Ingest documents in the background.

    Args:
        file_paths: Paths to files to ingest.
        chunking_strategy: Optional chunking strategy override.
        chunk_size: Optional chunk size override.
        chunk_overlap: Optional chunk overlap override.

    Returns:
        Dict with ingestion results.
    """
    from app.config.settings import ChunkingStrategy
    from app.services.dependencies import get_rag_service

    start = time.time()
    service = get_rag_service()
    strategy = ChunkingStrategy(chunking_strategy) if chunking_strategy else None

    try:
        result = service.upload_and_ingest(
            file_paths=file_paths,
            chunking_strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        elapsed = time.time() - start
        logger.info(
            "background_ingestion_completed",
            files=len(file_paths),
            total_chunks=result.total_chunks,
            duration_s=round(elapsed, 2),
        )
        return {
            "status": "completed",
            "total_chunks": result.total_chunks,
            "documents": len(result.documents),
            "duration_s": round(elapsed, 2),
        }
    except Exception as exc:
        elapsed = time.time() - start
        logger.error(
            "background_ingestion_failed",
            files=len(file_paths),
            error=str(exc),
            duration_s=round(elapsed, 2),
        )
        return {
            "status": "failed",
            "error": str(exc),
            "duration_s": round(elapsed, 2),
        }


def cleanup_expired_sessions_background(max_age_hours: int = 168) -> dict:
    """Clean up expired chat sessions in the background.

    Args:
        max_age_hours: Maximum session age in hours.

    Returns:
        Dict with cleanup results.
    """
    from app.memory.chat_store import cleanup_expired_sessions
    from app.models.database import _get_session_factory

    SessionLocal = _get_session_factory()
    db = SessionLocal()
    try:
        count = cleanup_expired_sessions(db, max_age_hours=max_age_hours)
        return {"status": "completed", "sessions_deleted": count}
    except Exception as exc:
        logger.error("session_cleanup_failed", error=str(exc))
        return {"status": "failed", "error": str(exc)}
    finally:
        db.close()


def cleanup_refresh_tokens_background(max_age_days: int = 30) -> dict:
    """Clean up expired refresh tokens in the background.

    Args:
        max_age_days: Maximum token age in days.

    Returns:
        Dict with cleanup results.
    """
    from datetime import datetime, timedelta, timezone

    from app.models.database import _get_session_factory
    from app.models.user import RefreshToken

    SessionLocal = _get_session_factory()
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        expired = (
            db.query(RefreshToken)
            .filter(RefreshToken.created_at < cutoff)
            .all()
        )
        count = len(expired)
        for token in expired:
            db.delete(token)
        db.commit()
        if count > 0:
            logger.info("refresh_tokens_cleaned", count=count)
        return {"status": "completed", "tokens_deleted": count}
    except Exception as exc:
        logger.error("token_cleanup_failed", error=str(exc))
        return {"status": "failed", "error": str(exc)}
    finally:
        db.close()
