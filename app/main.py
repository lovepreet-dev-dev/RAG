"""FastAPI application entry point.

Creates the application with lifespan events, CORS, global exception
handling, and the API router.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth import router as auth_router
from app.api.settings import router as settings_router
from app.api.chat_history import router as chat_history_router
from app.api.routes import router
from app.config.constants import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.config.settings import get_settings
from app.middleware.security import RateLimitMiddleware, SecurityHeadersMiddleware, RequestTracingMiddleware
from app.models.database import init_db
from app.utils.logging import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events.

    Args:
        app: The FastAPI application instance.
    """
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger(__name__)

    # Initialize database tables
    init_db()
    logger.info("database_initialized")

    # Seed default admin user if auth is disabled
    if not settings.enable_auth:
        _seed_default_admin()

    # Start background job queue
    from app.jobs import get_job_queue
    get_job_queue()
    logger.info("background_job_queue_started")

    issues = settings.validate_for_production()
    if issues:
        for issue in issues:
            logger.warning("production_issue", detail=issue)

    logger.info(
        "starting_rag_application",
        llm_provider=settings.llm_provider.value,
        vector_store=settings.vector_store.value,
        embedding_model=settings.embedding_model.value,
        version=APP_VERSION,
        auth_enabled=settings.enable_auth,
    )
    yield
    # Graceful shutdown: stop background job queue
    from app.jobs import get_job_queue
    get_job_queue().stop()
    logger.info("shutting_down_rag_application")


def _seed_default_admin():
    """Create a default admin user when auth is disabled."""
    from sqlalchemy.orm import Session

    from app.models.database import _get_session_factory
    from app.models.user import Role, User
    from app.services.auth_service import create_user

    try:
        SessionLocal = _get_session_factory()
        db = SessionLocal()
        try:
            existing = db.query(User).filter(User.email == "admin@example.com").first()
            if not existing:
                create_user(
                    db=db,
                    email="admin@example.com",
                    name="Admin",
                    password="admin",
                    role=Role.ADMIN,
                )
                get_logger(__name__).info("default_admin_seeded", email="admin@example.com")
        finally:
            db.close()
    except Exception as e:
        get_logger(__name__).warning("admin_seed_failed", error=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured ``FastAPI`` instance.
    """
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    settings = get_settings()
    allowed_origins = settings.cors_origins or ["*"]
    has_wildcard = "*" in allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=not has_wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if has_wildcard:
        logger = get_logger(__name__)
        logger.warning("cors_wildcard_with_credentials_disabled")

    # Security middleware
    is_production = not settings.debug
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=is_production)
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window,
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all exception handler for unhandled errors.

        Args:
            request: The incoming request.
            exc: The unhandled exception.

        Returns:
            JSON error response with 500 status.
        """
        logger = get_logger(__name__)
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again later."},
        )

    # Include routers
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(settings_router, prefix="/api/v1")
    app.include_router(chat_history_router, prefix="/api/v1")
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
