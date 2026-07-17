"""Settings API endpoints.

Handles reading and updating application settings stored in the database.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_role
from app.config.settings import get_settings
from app.models.schemas import ErrorResponse
from app.models.user import Role, SystemSetting, User
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/settings", tags=["Settings"])


class SettingsResponse(BaseModel):
    """Application settings response."""

    llm_provider: str = "openai"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    embedding_model: str = "bge-small"
    vector_store: str = "faiss"
    enable_hybrid_search: bool = False
    enable_reranking: bool = False
    similarity_top_k: int = 10
    hybrid_top_k: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 128
    max_context_tokens: int = 3000
    score_threshold: float = 0.3
    rerank_top_k: int = 5
    temperature: float = 0.3
    enable_answer_validation: bool = False


class SettingsUpdateRequest(BaseModel):
    """Settings update request (partial)."""

    llm_provider: Optional[str] = None
    openai_model: Optional[str] = None
    anthropic_model: Optional[str] = None
    openrouter_model: Optional[str] = None
    embedding_model: Optional[str] = None
    vector_store: Optional[str] = None
    enable_hybrid_search: Optional[bool] = None
    enable_reranking: Optional[bool] = None
    similarity_top_k: Optional[int] = Field(None, ge=1, le=100)
    hybrid_top_k: Optional[int] = Field(None, ge=1, le=100)
    chunk_size: Optional[int] = Field(None, ge=50, le=8192)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=4096)
    max_context_tokens: Optional[int] = Field(None, ge=100, le=100000)
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    rerank_top_k: Optional[int] = Field(None, ge=1, le=50)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    enable_answer_validation: Optional[bool] = None


# Default settings values (fallback when not in DB)
_DEFAULTS = {
    "llm_provider": "openai",
    "openai_model": "gpt-4o-mini",
    "anthropic_model": "claude-3-5-sonnet-20241022",
    "openrouter_model": "meta-llama/llama-3.1-8b-instruct:free",
    "embedding_model": "bge-small",
    "vector_store": "faiss",
    "enable_hybrid_search": "true",
    "enable_reranking": "false",
    "similarity_top_k": "10",
    "hybrid_top_k": "10",
    "chunk_size": "512",
    "chunk_overlap": "128",
    "max_context_tokens": "3000",
    "score_threshold": "0.3",
    "rerank_top_k": "5",
    "temperature": "0.3",
    "enable_answer_validation": "true",
}


def _get_setting_value(db: Session, key: str) -> str:
    """Get a setting value from the database, falling back to defaults."""
    record = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if record:
        return record.value
    return _DEFAULTS.get(key, "")


def _set_setting_value(
    db: Session, key: str, value: str, user_id: Optional[str] = None
) -> None:
    """Set a setting value in the database."""
    record = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if record:
        record.value = value
        record.updated_by = user_id
    else:
        record = SystemSetting(
            key=key,
            value=value,
            updated_by=user_id,
        )
        db.add(record)


def _parse_bool(value: str) -> bool:
    """Parse a string to boolean."""
    return value.lower() in ("true", "1", "yes")


@router.get(
    "",
    response_model=SettingsResponse,
    summary="Get application settings",
)
def get_settings_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve all application settings from the database."""
    return SettingsResponse(
        llm_provider=_get_setting_value(db, "llm_provider"),
        openai_model=_get_setting_value(db, "openai_model"),
        anthropic_model=_get_setting_value(db, "anthropic_model"),
        openrouter_model=_get_setting_value(db, "openrouter_model"),
        embedding_model=_get_setting_value(db, "embedding_model"),
        vector_store=_get_setting_value(db, "vector_store"),
        enable_hybrid_search=_parse_bool(_get_setting_value(db, "enable_hybrid_search")),
        enable_reranking=_parse_bool(_get_setting_value(db, "enable_reranking")),
        similarity_top_k=int(_get_setting_value(db, "similarity_top_k")),
        hybrid_top_k=int(_get_setting_value(db, "hybrid_top_k")),
        chunk_size=int(_get_setting_value(db, "chunk_size")),
        chunk_overlap=int(_get_setting_value(db, "chunk_overlap")),
        max_context_tokens=int(_get_setting_value(db, "max_context_tokens")),
        score_threshold=float(_get_setting_value(db, "score_threshold")),
        rerank_top_k=int(_get_setting_value(db, "rerank_top_k")),
        temperature=float(_get_setting_value(db, "temperature")),
        enable_answer_validation=_parse_bool(
            _get_setting_value(db, "enable_answer_validation")
        ),
    )


@router.put(
    "",
    response_model=SettingsResponse,
    summary="Update application settings",
    responses={403: {"model": ErrorResponse}},
)
def update_settings_endpoint(
    body: SettingsUpdateRequest,
    current_user: User = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
):
    """Update application settings (admin only)."""
    update_data = body.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        if value is not None:
            if isinstance(value, bool):
                _set_setting_value(db, key, str(value).lower(), current_user.id)
            else:
                _set_setting_value(db, key, str(value), current_user.id)

    db.commit()

    logger.info(
        "settings_updated",
        user_id=current_user.id,
        keys=list(update_data.keys()),
    )

    # Refresh the RAG service to pick up new settings
    try:
        from app.services.dependencies import get_rag_service
        # Collect all current DB settings as overrides
        all_overrides = {}
        for key in _DEFAULTS:
            all_overrides[key] = _get_setting_value(db, key)
        get_rag_service().refresh_settings(db_overrides=all_overrides)
    except Exception as exc:
        logger.warning("settings_refresh_failed", error=str(exc))

    # Return updated settings
    return get_settings_endpoint(current_user=current_user, db=db)
