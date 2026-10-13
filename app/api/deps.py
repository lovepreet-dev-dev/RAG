"""FastAPI dependencies for authentication and authorization.

Provides get_current_user, require_role, and optional auth dependencies.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.models.database import get_db
from app.models.user import Role, User
from app.services.auth_service import decode_token, get_user_by_id
from app.utils.logging import get_logger

logger = get_logger(__name__)

# HTTP Bearer scheme for JWT tokens
security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Extract and validate the current user from JWT token.

    Raises HTTPException 401 if:
    - No token provided
    - Token is invalid or expired
    - User not found or deactivated
    """
    settings = get_settings()

    # If auth is disabled, return a default admin user
    if not settings.enable_auth:
        return _get_or_create_default_user(db)

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return user


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Like get_current_user but returns None instead of raising.

    Useful for endpoints that behave differently for authenticated vs anonymous users.
    """
    settings = get_settings()

    if not settings.enable_auth:
        return _get_or_create_default_user(db)

    if credentials is None:
        return None

    token = credentials.credentials
    payload = decode_token(token)

    if payload is None or payload.get("type") != "access":
        return None

    user_id = payload.get("sub")
    if user_id is None:
        return None

    user = get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        return None

    return user


def require_role(*roles: Role):
    """Dependency factory that requires the user to have one of the specified roles.

    Usage:
        @router.get("/admin-only")
        def admin_endpoint(user: User = Depends(require_role(Role.ADMIN))):
            ...
    """

    def _check_role(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_role(*roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {', '.join(r.value for r in roles)}",
            )
        return current_user

    return _check_role


def _get_or_create_default_user(db: Session) -> User:
    """Get or create a default admin user when auth is disabled.

    This ensures the application works without authentication configured.
    """
    from app.services.auth_service import create_user

    default_email = "admin@example.com"

    user = db.query(User).filter(User.email == default_email).first()
    if user is None:
        try:
            user = create_user(
                db=db,
                email=default_email,
                name="Admin",
                password="admin",  # noqa: S106 - default dev password
                role=Role.ADMIN,
            )
            logger.info("default_admin_user_created", email=default_email)
        except ValueError:
            user = db.query(User).filter(User.email == default_email).first()

    return user
