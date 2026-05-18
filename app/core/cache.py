import hashlib
import json
from redis import Redis
from redis.exceptions import RedisError
from app.core.config import get_settings
from app.core.logger import app_logger

settings = get_settings()

# Cache key prefix to avoid collisions
CACHE_PREFIX = "neurorag:query:"


class CacheManager:
    """
    Manages Redis caching for RAG query results.

    Features:
    - SHA256 query hashing for consistent cache keys
    - Automatic TTL expiration
    - Graceful fallback — cache errors never break the pipeline
    - Cache hit/miss logging for monitoring
    """

    def __init__(self):
        """Initialize Redis connection. Fails gracefully if Redis unavailable."""
        self.client = None
        self.enabled = settings.cache_enabled

        if not self.enabled:
            app_logger.info("Cache disabled via config")
            return

        if not settings.redis_url:
            app_logger.warning("REDIS_URL not set — caching disabled")
            self.enabled = False
            return

        try:
            app_logger.info("Connecting to Redis...")
            self.client = Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self.client.ping()
            app_logger.info("Redis connected successfully")
        except Exception as e:
            app_logger.warning(f"Redis connection failed — caching disabled. Error: {e}")
            self.client = None
            self.enabled = False

    def _make_key(self, query: str) -> str:
        """
        Generate a consistent cache key from query string.
        Normalizes case and whitespace before hashing.

        Args:
            query: User query string

        Returns:
            Cache key string
        """
        normalized = query.strip().lower()
        query_hash = hashlib.sha256(normalized.encode()).hexdigest()
        return f"{CACHE_PREFIX}{query_hash}"

    def get(self, query: str) -> dict | None:
        """
        Retrieve cached result for a query.

        Args:
            query: User query string

        Returns:
            Cached result dict or None if not found
        """
        if not self.enabled or not self.client:
            return None

        try:
            key = self._make_key(query)
            cached = self.client.get(key)

            if cached:
                app_logger.info(f"Cache HIT for query: {query[:50]}...")
                return json.loads(cached)

            app_logger.info(f"Cache MISS for query: {query[:50]}...")
            return None

        except Exception as e:
            app_logger.warning(f"Cache get failed — proceeding without cache. Error: {e}")
            return None

    def set(self, query: str, result: dict) -> None:
        """
        Store query result in cache.

        Args:
            query: User query string
            result: Result dict to cache
        """
        if not self.enabled or not self.client:
            return

        try:
            key = self._make_key(query)
            # Exclude retrieved_chunks_text from cache — too large
            cacheable = {k: v for k, v in result.items()
                        if k != "retrieved_chunks_text"}
            self.client.setex(
                key,
                settings.cache_ttl,
                json.dumps(cacheable)
            )
            app_logger.info(
                f"Cached result for query: {query[:50]}... "
                f"(TTL: {settings.cache_ttl}s)"
            )
        except Exception as e:
            app_logger.warning(f"Cache set failed — result not cached. Error: {e}")

    def invalidate(self, query: str) -> None:
        """
        Delete cached result for a specific query.

        Args:
            query: User query string
        """
        if not self.enabled or not self.client:
            return

        try:
            key = self._make_key(query)
            self.client.delete(key)
            app_logger.info(f"Cache invalidated for query: {query[:50]}...")
        except Exception as e:
            app_logger.warning(f"Cache invalidation failed. Error: {e}")

    def flush(self) -> int:
        """
        Clear all NeuroRAG cache entries.

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.client:
            return 0

        try:
            pattern = f"{CACHE_PREFIX}*"
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                app_logger.info(f"Flushed {deleted} cache entries")
                return deleted
            return 0
        except Exception as e:
            app_logger.warning(f"Cache flush failed. Error: {e}")
            return 0

    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats or empty dict if unavailable
        """
        if not self.enabled or not self.client:
            return {"enabled": False, "connected": False}

        try:
            pattern = f"{CACHE_PREFIX}*"
            keys = self.client.keys(pattern)
            return {
                "enabled": True,
                "connected": True,
                "cached_queries": len(keys),
                "ttl_seconds": settings.cache_ttl,
            }
        except Exception as e:
            app_logger.warning(f"Cache stats failed. Error: {e}")
            return {"enabled": True, "connected": False}