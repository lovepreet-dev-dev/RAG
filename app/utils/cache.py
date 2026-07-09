"""Simple in-memory cache with TTL support.

Provides a thread-safe cache for frequently accessed data.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, TypeVar

from app.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class TTLCache:
    """Thread-safe cache with time-to-live expiration.

    Args:
        max_size: Maximum number of entries.
        default_ttl: Default TTL in seconds.
    """

    def __init__(self, max_size: int = 100, default_ttl: int = 60):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Get a value from cache, returning None if expired or missing."""
        with self._lock:
            if key not in self._cache:
                return None
            value, expires_at = self._cache[key]
            if time.time() > expires_at:
                del self._cache[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in cache with optional TTL override."""
        with self._lock:
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Remove oldest entry
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            expires_at = time.time() + (ttl or self._default_ttl)
            self._cache[key] = (value, expires_at)

    def invalidate(self, key: str) -> bool:
        """Remove a key from cache. Returns True if key existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
            }


# Global cache instance
_cache: TTLCache | None = None


def get_cache() -> TTLCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = TTLCache(max_size=200, default_ttl=30)
    return _cache


def cached(ttl: int | None = None, key_prefix: str = ""):
    """Decorator for caching function results.

    Args:
        ttl: Time-to-live in seconds.
        key_prefix: Prefix for cache key.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            cache = get_cache()
            # Build cache key from function name and arguments
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
