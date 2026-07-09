"""Chat history API endpoints.

Handles CRUD operations for chat sessions and messages.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.schemas import ErrorResponse
from app.models.user import ChatMessage, ChatSession, User
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat History"])


# ── Request/Response Models ─────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    """Create a new chat session."""

    title: str = Field(default="New Chat", max_length=255)


class UpdateSessionRequest(BaseModel):
    """Update a chat session."""

    title: Optional[str] = Field(None, max_length=255)
    is_pinned: Optional[bool] = None


class MessageResponse(BaseModel):
    """Chat message response."""

    id: str
    role: str
    content: str
    query_id: Optional[str] = None
    sources: Optional[list[dict]] = None
    confidence_score: Optional[float] = None
    confidence_level: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    created_at: str


class SessionResponse(BaseModel):
    """Chat session response."""

    id: str
    title: str
    is_pinned: bool
    message_count: int
    created_at: str
    updated_at: str


class SessionDetailResponse(BaseModel):
    """Chat session with messages."""

    id: str
    title: str
    is_pinned: bool
    messages: list[MessageResponse]
    created_at: str
    updated_at: str


class SessionListResponse(BaseModel):
    """List of chat sessions."""

    sessions: list[SessionResponse]
    total_count: int


class AddMessageRequest(BaseModel):
    """Add a message to a session."""

    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    query_id: Optional[str] = None
    sources: Optional[list[dict]] = None
    confidence_score: Optional[float] = None
    confidence_level: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None


# ── Session Endpoints ───────────────────────────────────────────────────


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List chat sessions",
)
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List all chat sessions for the current user."""
    query = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.is_pinned.desc(), ChatSession.updated_at.desc())
    )

    total = query.count()
    sessions = query.offset(offset).limit(limit).all()

    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=s.id,
                title=s.title,
                is_pinned=s.is_pinned,
                message_count=len(s.messages),
                created_at=s.created_at.isoformat() if s.created_at else "",
                updated_at=s.updated_at.isoformat() if s.updated_at else "",
            )
            for s in sessions
        ],
        total_count=total,
    )


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a chat session",
)
def create_session(
    body: CreateSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new chat session."""
    session = ChatSession(
        user_id=current_user.id,
        title=body.title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    logger.info("chat_session_created", session_id=session.id, user_id=current_user.id)

    return SessionResponse(
        id=session.id,
        title=session.title,
        is_pinned=session.is_pinned,
        message_count=0,
        created_at=session.created_at.isoformat() if session.created_at else "",
        updated_at=session.updated_at.isoformat() if session.updated_at else "",
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a chat session with all messages."""
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    messages = sorted(session.messages, key=lambda m: m.created_at or "")

    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        is_pinned=session.is_pinned,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                query_id=m.query_id,
                sources=json.loads(m.sources_json) if m.sources_json else None,
                confidence_score=m.confidence_score,
                confidence_level=m.confidence_level,
                model_used=m.model_used,
                tokens_used=m.tokens_used,
                latency_ms=m.latency_ms,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in messages
        ],
        created_at=session.created_at.isoformat() if session.created_at else "",
        updated_at=session.updated_at.isoformat() if session.updated_at else "",
    )


@router.put(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}},
)
def update_session(
    session_id: str,
    body: UpdateSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update a chat session (rename, pin/unpin)."""
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if body.title is not None:
        session.title = body.title
    if body.is_pinned is not None:
        session.is_pinned = body.is_pinned

    db.commit()
    db.refresh(session)

    return SessionResponse(
        id=session.id,
        title=session.title,
        is_pinned=session.is_pinned,
        message_count=len(session.messages),
        created_at=session.created_at.isoformat() if session.created_at else "",
        updated_at=session.updated_at.isoformat() if session.updated_at else "",
    )


@router.delete(
    "/sessions/{session_id}",
    responses={404: {"model": ErrorResponse}},
)
def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a chat session and all its messages."""
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    db.delete(session)
    db.commit()

    logger.info("chat_session_deleted", session_id=session_id, user_id=current_user.id)
    return {"message": "Session deleted"}


# ── Message Endpoints ───────────────────────────────────────────────────


@router.post(
    "/sessions/{session_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}},
)
def add_message(
    session_id: str,
    body: AddMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a message to a chat session."""
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    message = ChatMessage(
        session_id=session_id,
        role=body.role,
        content=body.content,
        query_id=body.query_id,
        sources_json=json.dumps(body.sources) if body.sources else None,
        confidence_score=body.confidence_score,
        confidence_level=body.confidence_level,
        model_used=body.model_used,
        tokens_used=body.tokens_used,
        latency_ms=body.latency_ms,
    )
    db.add(message)

    # Update session timestamp
    from datetime import datetime, timezone

    session.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(message)

    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        query_id=message.query_id,
        sources=json.loads(message.sources_json) if message.sources_json else None,
        confidence_score=message.confidence_score,
        confidence_level=message.confidence_level,
        model_used=message.model_used,
        tokens_used=message.tokens_used,
        latency_ms=message.latency_ms,
        created_at=message.created_at.isoformat() if message.created_at else "",
    )
