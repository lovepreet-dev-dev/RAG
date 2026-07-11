"""Authentication API endpoints.

Handles login, registration, token refresh, password changes, and user management.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_current_user_optional, get_db, require_role
from app.config.settings import get_settings
from app.models.schemas import (
    ChangePasswordRequest,
    ErrorResponse,
    LoginRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
    UpdateUserRequest,
    UserListResponse,
    UserResponse,
)
from app.models.user import Role, User
from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    create_user,
    get_user_by_email,
    is_refresh_token_valid,
    revoke_all_refresh_tokens,
    revoke_refresh_token,
    update_user,
    user_to_response,
)
from app.utils.audit import log_audit
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# In-memory failed login tracking: {email: (count, first_failure_time)}
_FAILED_LOGINS: dict[str, tuple[int, float]] = {}
_MAX_LOGIN_ATTEMPTS = 5
_LOCKOUT_SECONDS = 900  # 15 minutes


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={401: {"model": ErrorResponse}, 423: {"model": ErrorResponse}},
)
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return JWT tokens.

    Returns 423 Locked after too many failed attempts.
    """
    # Check account lockout
    now = time.time()
    if body.email in _FAILED_LOGINS:
        count, first_at = _FAILED_LOGINS[body.email]
        if count >= _MAX_LOGIN_ATTEMPTS and (now - first_at) < _LOCKOUT_SECONDS:
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to too many failed login attempts. Try again later.",
            )
        elif (now - first_at) >= _LOCKOUT_SECONDS:
            _FAILED_LOGINS.pop(body.email, None)

    user = authenticate_user(db, body.email, body.password)
    if user is None:
        # Track failed attempt
        if body.email in _FAILED_LOGINS:
            count, first_at = _FAILED_LOGINS[body.email]
            _FAILED_LOGINS[body.email] = (count + 1, first_at)
        else:
            _FAILED_LOGINS[body.email] = (1, now)

        logger.warning(
            "login_failed",
            email=body.email,
            ip=request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Clear failed attempts on success
    _FAILED_LOGINS.pop(body.email, None)

    # Capture values before commit (which expires the identity)
    user_id = user.id
    user_role = user.role.value

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    # Generate tokens
    access_token = create_access_token(user_id, user_role)
    refresh_token = create_refresh_token(user_id, db)

    settings = get_settings()

    log_audit(
        db,
        user_id=user.id,
        action="login",
        resource_type="user",
        resource_id=user.id,
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", ""),
    )

    logger.info(
        "user_login",
        user_id=user.id,
        email=user.email,
        ip=request.client.host if request.client else "unknown",
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
def register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """Register a new user account.

    When auth is enabled, only admins can assign roles other than 'employee'.
    Self-registration (without auth) is always limited to 'employee' role.
    """
    settings = get_settings()

    # Determine allowed role
    if settings.enable_auth and current_user and current_user.has_role(Role.ADMIN):
        # Admin can assign any role
        try:
            role = Role(body.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {body.role}",
            )
    else:
        # Non-admin or unauthenticated: force employee role
        role = Role.EMPLOYEE

    try:
        user = create_user(
            db=db,
            email=body.email,
            name=body.name,
            password=body.password,
            role=role,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    log_audit(
        db,
        user_id=user.id,
        action="register",
        resource_type="user",
        resource_id=user.id,
        details=f"role={role.value}",
    )

    return user_to_response(user)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    responses={401: {"model": ErrorResponse}},
)
def refresh_token(body: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Refresh an access token using a refresh token."""
    settings = get_settings()

    # Verify the refresh token exists and is valid
    if not is_refresh_token_valid(db, body.refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Decode to get user_id
    from app.services.auth_service import decode_token

    payload = decode_token(body.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    from app.services.auth_service import get_user_by_id

    user = get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )

    # Capture values before operations that commit
    user_id_val = user.id
    user_role_val = user.role.value

    # Revoke old refresh token (rotation)
    revoke_refresh_token(db, body.refresh_token)

    # Generate new tokens
    access_token = create_access_token(user_id_val, user_role_val)
    refresh_token = create_refresh_token(user_id_val, db)

    log_audit(
        db,
        user_id=user.id,
        action="token_refresh",
        resource_type="user",
        resource_id=user.id,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user information."""
    return user_to_response(current_user)


@router.post(
    "/logout",
    responses={200: {"model": dict}},
)
def logout(
    body: RefreshTokenRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Logout by revoking the refresh token.

    Verifies the token belongs to the current user before revoking.
    """
    from app.services.auth_service import decode_token

    payload = decode_token(body.refresh_token)
    if payload is None or payload.get("sub") != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid refresh token",
        )
    revoke_refresh_token(db, body.refresh_token)
    log_audit(db, user_id=current_user.id, action="logout", resource_type="user", resource_id=current_user.id)
    logger.info("user_logout", user_id=current_user.id)
    return {"message": "Successfully logged out"}


@router.post(
    "/change-password",
    responses={400: {"model": ErrorResponse}},
)
def change_password(
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change the current user's password."""
    from app.services.auth_service import verify_password, hash_password

    # Verify current password
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password
    current_user.hashed_password = hash_password(body.new_password)
    db.commit()

    # Revoke all refresh tokens for security
    revoke_all_refresh_tokens(db, current_user.id)

    log_audit(db, user_id=current_user.id, action="change_password", resource_type="user", resource_id=current_user.id)

    logger.info("password_changed", user_id=current_user.id)
    return {"message": "Password changed successfully"}


# ── Admin User Management Endpoints ────────────────────────────────────


@router.get(
    "/users",
    response_model=UserListResponse,
)
def list_users(
    current_user: User = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
):
    """List all users (admin only)."""
    users = db.query(User).all()
    return UserListResponse(
        users=[user_to_response(u) for u in users],
        total_count=len(users),
    )


@router.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}},
)
def create_user_endpoint(
    body: RegisterRequest,
    current_user: User = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
):
    """Create a new user (admin only)."""
    try:
        role = Role(body.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {body.role}",
        )

    try:
        user = create_user(
            db=db,
            email=body.email,
            name=body.name,
            password=body.password,
            role=role,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    log_audit(
        db,
        user_id=current_user.id,
        action="create_user",
        resource_type="user",
        resource_id=user.id,
        details=f"email={body.email}, role={body.role}",
    )

    return user_to_response(user)


@router.put(
    "/users/{user_id}",
    response_model=UserResponse,
    responses={404: {"model": ErrorResponse}},
)
def update_user_endpoint(
    user_id: str,
    body: UpdateUserRequest,
    current_user: User = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
):
    """Update a user (admin only).

    Prevents self-deactivation, self-demotion, and removal of the last admin.
    """
    role = None
    if body.role is not None:
        try:
            role = Role(body.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {body.role}",
            )

    user = update_user(
        db=db,
        user_id=user_id,
        name=body.name,
        role=role,
        is_active=body.is_active,
        password=body.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Prevent self-deactivation and self-demotion
    if user.id == current_user.id:
        if body.is_active is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account",
            )
        if role is not None and role != Role.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote your own account",
            )

    # Last-admin protection: prevent demoting or deactivating the last admin
    if (body.is_active is False or (role is not None and role != Role.ADMIN)):
        admin_count = (
            db.query(User)
            .filter(User.role == Role.ADMIN, User.is_active == True, User.id != user_id)
            .count()
        )
        if admin_count < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate or demote the last admin user",
            )

    log_audit(
        db,
        user_id=current_user.id,
        action="update_user",
        resource_type="user",
        resource_id=user_id,
        details=f"name={body.name}, role={body.role}, is_active={body.is_active}",
    )

    return user_to_response(user)


@router.delete(
    "/users/{user_id}",
    responses={404: {"model": ErrorResponse}},
)
def delete_user_endpoint(
    user_id: str,
    current_user: User = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
):
    """Delete a user (admin only).

    Prevents self-deletion and deletion of the last admin.
    """
    from app.services.auth_service import get_user_by_id

    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Prevent self-deletion
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    # Last-admin protection
    if user.role == Role.ADMIN:
        admin_count = (
            db.query(User)
            .filter(User.role == Role.ADMIN, User.is_active == True, User.id != user_id)
            .count()
        )
        if admin_count < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the last admin user",
            )

    db.delete(user)
    db.commit()

    log_audit(
        db,
        user_id=current_user.id,
        action="delete_user",
        resource_type="user",
        resource_id=user_id,
        details=f"email={user.email}",
    )

    logger.info("user_deleted", user_id=user_id, deleted_by=current_user.id)
    return {"message": "User deleted successfully"}


# ── Password Reset Endpoints ────────────────────────────────────────────


class ForgotPasswordRequest(BaseModel):
    """Forgot password request."""

    email: str = Field(..., min_length=3, max_length=255)


class ResetPasswordRequest(BaseModel):
    """Reset password request with token."""

    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)


@router.post(
    "/forgot-password",
    responses={200: {"model": dict}},
)
def forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Request a password reset token.

    Always returns success to prevent user enumeration.
    The token is logged for development purposes.
    """
    import secrets
    from datetime import datetime, timedelta, timezone

    from app.models.user import PasswordResetToken

    user = get_user_by_email(db, body.email)

    # Always return success to prevent user enumeration
    if user is None:
        logger.info("password_reset_requested_nonexistent", email=body.email)
        return {"message": "If an account exists, a reset link has been sent."}

    # Generate reset token
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    reset_token = PasswordResetToken(
        user_id=user.id,
        token=token,
        expires_at=expires_at,
    )
    db.add(reset_token)
    db.commit()

    log_audit(
        db,
        user_id=user.id,
        action="password_reset_requested",
        resource_type="user",
        resource_id=user.id,
    )

    logger.info(
        "password_reset_token_generated",
        user_id=user.id,
        email=user.email,
    )

    return {"message": "If an account exists, a reset link has been sent."}


@router.post(
    "/reset-password",
    responses={400: {"model": ErrorResponse}},
)
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Reset password using a valid reset token."""
    from datetime import datetime, timezone

    from app.models.user import PasswordResetToken
    from app.services.auth_service import hash_password, revoke_all_refresh_tokens

    reset_record = (
        db.query(PasswordResetToken)
        .filter(
            PasswordResetToken.token == body.token,
            PasswordResetToken.used == False,
        )
        .first()
    )

    if reset_record is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Check expiration
    if reset_record.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired",
        )

    # Get user
    user = get_user_by_id(db, reset_record.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not found or deactivated",
        )

    # Update password
    user.hashed_password = hash_password(body.new_password)
    reset_record.used = True
    db.commit()

    # Revoke all refresh tokens for security
    revoke_all_refresh_tokens(db, user.id)

    log_audit(
        db,
        user_id=user.id,
        action="password_reset_completed",
        resource_type="user",
        resource_id=user.id,
    )

    logger.info("password_reset_completed", user_id=user.id)
    return {"message": "Password reset successfully"}
