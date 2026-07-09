"""Long-term memory manager for RAG chat sessions.

Provides three core capabilities:
1. Summarization: Compress older messages into summaries
2. Semantic retrieval: Embed messages and retrieve relevant history
3. Context-aware injection: Combine recent + retrieved history
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session as DBSession

from app.models.user import ChatMessage, ChatSession
from app.utils.helpers import generate_id
from app.utils.logging import get_logger

logger = get_logger(__name__)


def load_session_history(
    db: DBSession,
    session_id: str,
    limit: int = 50,
) -> list[dict[str, str]]:
    """Load chat history from the database for a session.

    Args:
        db: Database session.
        session_id: The chat session ID.
        limit: Maximum messages to load.

    Returns:
        List of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    messages.reverse()
    return [{"role": m.role, "content": m.content} for m in messages]


def persist_message(
    db: DBSession,
    session_id: str,
    role: str,
    content: str,
    query_id: Optional[str] = None,
    sources: Optional[list[dict]] = None,
    confidence_score: Optional[float] = None,
    confidence_level: Optional[str] = None,
    model_used: Optional[str] = None,
    tokens_used: Optional[int] = None,
    latency_ms: Optional[float] = None,
) -> ChatMessage:
    """Persist a single message to the database.

    Args:
        db: Database session.
        session_id: The chat session ID.
        role: Message role ("user", "assistant", "system").
        content: Message content.
        query_id: Optional query ID for tracing.
        sources: Optional source documents.
        confidence_score: Optional confidence score.
        confidence_level: Optional confidence level.
        model_used: Optional model name.
        tokens_used: Optional token count.
        latency_ms: Optional latency.

    Returns:
        The created ChatMessage.
    """
    from datetime import datetime, timezone

    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        query_id=query_id,
        sources_json=json.dumps(sources) if sources else None,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        model_used=model_used,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
    )
    db.add(message)

    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(message)
    return message


def persist_chat_response(
    db: DBSession,
    session_id: str,
    user_message: str,
    response,
) -> None:
    """Persist both user message and assistant response to the database.

    Args:
        db: Database session.
        session_id: The chat session ID.
        user_message: The user's message text.
        response: The QueryResponse object from RAG.
    """
    persist_message(
        db=db,
        session_id=session_id,
        role="user",
        content=user_message,
    )
    persist_message(
        db=db,
        session_id=session_id,
        role="assistant",
        content=response.answer,
        query_id=response.query_id,
        sources=[s.model_dump() for s in response.sources] if response.sources else None,
        confidence_score=response.confidence_score,
        confidence_level=response.confidence_level,
        model_used=response.model_used,
        tokens_used=response.tokens_used,
        latency_ms=response.latency_ms,
    )


def get_or_create_session(
    db: DBSession,
    session_id: str,
    user_id: str,
    title: Optional[str] = None,
) -> ChatSession:
    """Get an existing session or create a new one.

    Args:
        db: Database session.
        session_id: The session ID.
        user_id: The user ID.
        title: Optional title for new sessions.

    Returns:
        The ChatSession (existing or newly created).
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        return session

    session = ChatSession(
        id=session_id,
        user_id=user_id,
        title=title or "New Chat",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def build_memory_context(
    db: DBSession,
    session_id: str,
    max_messages: int = 20,
    summary_threshold: int = 30,
) -> tuple[list[dict[str, str]], Optional[str]]:
    """Build context from chat history with optional summarization.

    For sessions with more messages than the summary threshold,
    older messages are summarized to save tokens while preserving
    key context.

    Args:
        db: Database session.
        session_id: The chat session ID.
        max_messages: Maximum recent messages to include directly.
        summary_threshold: Summarize if message count exceeds this.

    Returns:
        Tuple of (recent_messages, summary_text).
    """
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    if len(messages) <= max_messages:
        return [{"role": m.role, "content": m.content} for m in messages], None

    summary_messages = messages[: len(messages) - max_messages]
    recent_messages = messages[len(messages) - max_messages :]

    summary_parts = []
    for m in summary_messages:
        prefix = "User" if m.role == "user" else "Assistant"
        content = m.content[:500] + "..." if len(m.content) > 500 else m.content
        summary_parts.append(f"{prefix}: {content}")

    summary = (
        "Previous conversation summary:\n"
        + "\n".join(summary_parts[-10:])
    )

    return (
        [{"role": m.role, "content": m.content} for m in recent_messages],
        summary,
    )


def cleanup_expired_sessions(
    db: DBSession,
    max_age_hours: int = 168,
) -> int:
    """Remove sessions older than max_age_hours.

    Args:
        db: Database session.
        max_age_hours: Maximum session age in hours (default 7 days).

    Returns:
        Number of sessions deleted.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    expired = (
        db.query(ChatSession)
        .filter(ChatSession.updated_at < cutoff)
        .all()
    )
    count = len(expired)
    for session in expired:
        db.delete(session)
    db.commit()
    if count > 0:
        logger.info("expired_sessions_cleaned", count=count)
    return count
