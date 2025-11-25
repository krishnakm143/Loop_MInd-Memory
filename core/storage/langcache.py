"""LangCache semantic caching wrapper."""

from typing import Optional, Dict, Any
from langcache import LangCache
from api.config import settings
import structlog

logger = structlog.get_logger()


class LangCacheStore:
    """Wrapper for LangCache semantic caching operations."""

    def __init__(self):
        """Initialize LangCache client."""
        self.lang_cache: Optional[LangCache] = None
        self._connected = False

    def connect(self):
        """Connect to LangCache."""
        if not settings.enable_langcache:
            logger.debug("LangCache is disabled")
            return False

        if not settings.langcache_api_key:
            logger.warning("LangCache API key not set, semantic caching disabled")
            return False

        try:
            self.lang_cache = LangCache(
                server_url=settings.langcache_server_url,
                cache_id=settings.langcache_cache_id,
                api_key=settings.langcache_api_key,
            )
            self._connected = True
            logger.info("LangCache connected successfully")
            return True
        except Exception as e:
            logger.error("Failed to connect to LangCache", error=str(e))
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from LangCache."""
        if self.lang_cache:
            try:
                # LangCache context manager handles cleanup
                self.lang_cache = None
                self._connected = False
                logger.info("LangCache disconnected")
            except Exception as e:
                logger.error("Error disconnecting from LangCache", error=str(e))

    def search(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Search for cached response using semantic similarity.

        Args:
            prompt: Query prompt to search for

        Returns:
            Cached response if found, None otherwise
        """
        if not self._connected or not self.lang_cache:
            return None

        try:
            response = self.lang_cache.search(prompt=prompt)
            if response:
                logger.debug("LangCache hit", prompt_preview=prompt[:50])
                return response
            return None
        except Exception as e:
            logger.error("LangCache search failed", error=str(e))
            return None

    def set(self, prompt: str, response: Dict[str, Any]) -> bool:
        """Cache a query response.

        Args:
            prompt: Query prompt
            response: Response to cache (should be serializable)

        Returns:
            True if cached successfully
        """
        if not self._connected or not self.lang_cache:
            return False

        try:
            # Convert response dict to string for LangCache
            response_str = str(response) if not isinstance(response, str) else response
            result = self.lang_cache.set(prompt=prompt, response=response_str)
            if result:
                logger.debug("LangCache set successful", prompt_preview=prompt[:50])
                return True
            return False
        except Exception as e:
            logger.error("LangCache set failed", error=str(e))
            return False

    def health_check(self) -> bool:
        """Check if LangCache is healthy."""
        return self._connected and self.lang_cache is not None


# Global LangCache instance
_langcache_store: Optional[LangCacheStore] = None


def get_langcache_store() -> LangCacheStore:
    """Get or create LangCache store instance."""
    global _langcache_store
    if _langcache_store is None:
        _langcache_store = LangCacheStore()
        _langcache_store.connect()
    return _langcache_store
