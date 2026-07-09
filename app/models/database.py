"""Database configuration and session management.

Uses SQLAlchemy with SQLite (default) or PostgreSQL.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# Lazy-initialized engine and session factory
_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        database_url = settings.database_url

        connect_args = {}
        pool_kwargs = {
            "pool_pre_ping": True,
        }

        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        else:
            # PostgreSQL connection pooling
            pool_kwargs["pool_size"] = 10
            pool_kwargs["max_overflow"] = 20
            pool_kwargs["pool_timeout"] = 30
            pool_kwargs["pool_recycle"] = 1800

        _engine = create_engine(
            database_url,
            connect_args=connect_args,
            echo=settings.debug,
            **pool_kwargs,
        )

        # Enable WAL mode for SQLite
        if database_url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        logger.info("database_engine_created", database_url=database_url.split("@")[-1] if "@" in database_url else database_url)
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_get_engine(),
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a database session."""
    SessionLocal = _get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("database_tables_created")


def reset_db() -> None:
    """Drop and recreate all tables (for testing)."""
    global _engine, _SessionLocal
    engine = _get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    _SessionLocal = None
    logger.info("database_reset")
