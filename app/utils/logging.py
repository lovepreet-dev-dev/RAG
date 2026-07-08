"""Structured logging configuration using structlog.

Provides JSON-formatted, correlation-ID-aware logging with latency tracking.
Every module calls ``get_logger(__name__)`` to obtain a bound logger instance.
"""

from __future__ import annotations

import logging
import sys
import time
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog

F = TypeVar("F", bound=Callable[..., Any])

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the entire application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a bound logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A structlog bound logger.
    """
    if not _configured:
        setup_logging()
    return structlog.get_logger(name)


def log_latency(func: F) -> F:
    """Decorator that logs execution latency for sync and async functions.

    On success it emits an ``info`` log with ``latency_ms``.
    On failure it emits an ``error`` log and re-raises.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function with latency logging.
    """
    logger = get_logger(func.__module__)

    import asyncio
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                latency = (time.perf_counter() - start) * 1000
                logger.info(
                    "function_executed",
                    function=func.__qualname__,
                    latency_ms=round(latency, 2),
                    status="success",
                )
                return result
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                logger.error(
                    "function_failed",
                    function=func.__qualname__,
                    latency_ms=round(latency, 2),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise
        return async_wrapper  # type: ignore[return-value]
    else:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                latency = (time.perf_counter() - start) * 1000
                logger.info(
                    "function_executed",
                    function=func.__qualname__,
                    latency_ms=round(latency, 2),
                    status="success",
                )
                return result
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                logger.error(
                    "function_failed",
                    function=func.__qualname__,
                    latency_ms=round(latency, 2),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise
        return sync_wrapper  # type: ignore[return-value]
