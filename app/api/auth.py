"""Authentication API endpoints.

Handles login, registration, token refresh, password changes, and user management.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_role
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
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={401: {"model": ErrorResponse}},
)
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return JWT tokens."""
    user = authenticate_user(db, body.email, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    # Generate tokens
    access_token = create_access_token(user.id, user.role.value)
    refresh_token = create_refresh_token(user.id, db)

    settings = get_settings()

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
    responses={400: {"model": ErrorResponse}},
)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
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

    # Revoke old refresh token (rotation)
    revoke_refresh_token(db, body.refresh_token)

    # Generate new tokens
    access_token = create_access_token(user.id, user.role.value)
    refresh_token = create_refresh_token(user.id, db)

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
    """Logout by revoking the refresh token."""
    revoke_refresh_token(db, body.refresh_token)
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
    """Update a user (admin only)."""
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
    """Delete a user (admin only)."""
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

    db.delete(user)
    db.commit()

    logger.info("user_deleted", user_id=user_id, deleted_by=current_user.id)
    return {"message": "User deleted successfully"}
