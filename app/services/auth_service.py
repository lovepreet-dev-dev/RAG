"""Authentication service.

Handles JWT token creation/verification, password hashing,
user management, and session handling.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.models.user import RefreshToken, Role, User
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str, role: str) -> str:
    """Create a JWT access token."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.access_token_expire_minutes)

    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "exp": expire,
        "iat": now,
        "iss": "rag-engine",
    }

    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_refresh_token(user_id: str, db: Session) -> str:
    """Create a refresh token and store it in the database."""
    import uuid

    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.refresh_token_expire_days)

    # Generate the token (include jti for uniqueness)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": now,
        "jti": str(uuid.uuid4()),
        "iss": "rag-engine",
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)

    # Store in database
    db_token = RefreshToken(
        token=token,
        user_id=user_id,
        expires_at=expire,
    )
    db.add(db_token)
    db.commit()

    return token


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT token. Returns payload or None."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        return payload
    except JWTError as e:
        logger.debug("token_decode_failed", error=str(e))
        return None


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Fetch a user by email address."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    """Fetch a user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user with email and password."""
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(
    db: Session,
    email: str,
    name: str,
    password: str,
    role: Role = Role.EMPLOYEE,
) -> User:
    """Create a new user account."""
    # Check for existing user
    existing = get_user_by_email(db, email)
    if existing:
        raise ValueError(f"User with email {email} already exists")

    user = User(
        email=email,
        name=name,
        hashed_password=hash_password(password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info("user_created", user_id=user.id, email=email, role=role.value)
    return user


def update_user(
    db: Session,
    user_id: str,
    name: Optional[str] = None,
    role: Optional[Role] = None,
    is_active: Optional[bool] = None,
    password: Optional[str] = None,
) -> Optional[User]:
    """Update an existing user."""
    user = get_user_by_id(db, user_id)
    if not user:
        return None

    if name is not None:
        user.name = name
    if role is not None:
        user.role = role
    if is_active is not None:
        user.is_active = is_active
    if password is not None:
        user.hashed_password = hash_password(password)

    db.commit()
    db.refresh(user)
    return user


def revoke_all_refresh_tokens(db: Session, user_id: str) -> int:
    """Revoke all refresh tokens for a user. Returns count revoked."""
    count = (
        db.query(RefreshToken)
        .filter(
            RefreshToken.user_id == user_id,
            RefreshToken.is_revoked == False,
        )
        .update({"is_revoked": True})
    )
    db.commit()
    return count


def revoke_refresh_token(db: Session, token: str) -> bool:
    """Revoke a specific refresh token."""
    db_token = (
        db.query(RefreshToken)
        .filter(RefreshToken.token == token, RefreshToken.is_revoked == False)
        .first()
    )
    if not db_token:
        return False

    db_token.is_revoked = True
    db.commit()
    return True


def is_refresh_token_valid(db: Session, token: str) -> bool:
    """Check if a refresh token is valid and not revoked."""
    db_token = (
        db.query(RefreshToken)
        .filter(RefreshToken.token == token, RefreshToken.is_revoked == False)
        .first()
    )
    if not db_token:
        return False

    # Check expiration - handle both naive and aware datetimes
    expires_at = db_token.expires_at
    now = datetime.now(timezone.utc)
    if expires_at.tzinfo is None:
        # Make naive datetime timezone-aware (assume UTC)
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < now:
        db_token.is_revoked = True
        db.commit()
        return False

    return True


def user_to_response(user: User) -> dict:
    """Convert a User model to a UserResponse dict."""
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role.value,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }
