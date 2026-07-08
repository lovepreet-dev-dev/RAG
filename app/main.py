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

from app.api.routes import router
from app.config.constants import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.config.settings import get_settings
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
    )
    yield
    logger.info("shutting_down_rag_application")


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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
