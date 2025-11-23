"""Redis cache client wrapper."""

from typing import Optional, Any
import redis.asyncio as redis
from api.config import settings
import json


class CacheStore:
    """Wrapper for Redis cache operations."""

    def __init__(self):
        """Initialize Redis client."""
        self.client: Optional[redis.Redis] = None

    async def connect(self):
        """Connect to Redis."""
        if not self.client:
            # Build Redis URL with authentication if provided
            if settings.redis_username and settings.redis_password:
                # Construct URL with username and password
                redis_url = f"redis://{settings.redis_username}:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/0"
            elif settings.redis_password:
                # URL with password only
                redis_url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/0"
            else:
                # Use provided URL or default
                redis_url = settings.redis_url
            
            self.client = await redis.from_url(
                redis_url, encoding="utf-8", decode_responses=True
            )

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            self.client = None

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if not self.client:
            await self.connect()
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL."""
        if not self.client:
            await self.connect()
        try:
            serialized = json.dumps(value)
            if ttl:
                await self.client.setex(key, ttl, serialized)
            else:
                await self.client.set(key, serialized)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.client:
            await self.connect()
        try:
            await self.client.delete(key)
            return True
        except Exception:
            return False

    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            if not self.client:
                await self.connect()
            await self.client.ping()
            return True
        except Exception:
            return False

