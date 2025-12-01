"""
Performance Cache Layer.

Provides caching for expensive operations:
- Embedding vectors (avoid recomputing)
- Graph traversal results (avoid repeated queries)
- Sector classification results (avoid repeated inference)
- Query results (semantic caching via LangCache)

Uses Redis with configurable TTLs per cache type.
Target: Reduce p50 latency from ~150ms to <100ms.
"""

import hashlib
import json
import structlog
from typing import Any, Dict, List, Optional
from datetime import datetime
import redis.asyncio as redis

from api.config import settings

logger = structlog.get_logger(__name__)


# Cache TTL configuration (seconds)
CACHE_TTL = {
    "embedding": 3600,       # 1 hour - embeddings don't change
    "classification": 300,    # 5 minutes - classification results
    "graph_neighbors": 60,    # 1 minute - graph can change
    "query_results": 300,     # 5 minutes - query cache
}

# Cache key prefixes
CACHE_PREFIX = {
    "embedding": "emb:",
    "classification": "cls:",
    "graph_neighbors": "gn:",
    "query_results": "qr:",
}


def _hash_key(data: str) -> str:
    """Create a short hash for cache keys."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class PerformanceCache:
    """
    Performance-focused cache for retrieval operations.
    
    Caches:
    - Embedding vectors to avoid API calls
    - Classification results to avoid inference
    - Graph traversal results to reduce Neo4j queries
    - Full query results for repeated queries
    """
    
    def __init__(self):
        """Initialize cache client."""
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._enabled = True  # Can be disabled for testing
    
    async def _ensure_connected(self):
        """Ensure Redis connection is established."""
        if self._connected and self._client:
            return
        
        try:
            # Build Redis URL
            if settings.redis_username and settings.redis_password:
                redis_url = f"redis://{settings.redis_username}:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/0"
            elif settings.redis_password:
                redis_url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/0"
            else:
                redis_url = settings.redis_url
            
            self._client = await redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._client.ping()
            self._connected = True
            logger.info("Performance cache connected to Redis")
            
        except Exception as e:
            logger.warning("Performance cache connection failed", error=str(e))
            self._enabled = False
    
    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._connected = False
    
    # =========================================================================
    # Embedding Cache
    # =========================================================================
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        if not self._enabled:
            return None
        
        await self._ensure_connected()
        if not self._client:
            return None
        
        try:
            key = f"{CACHE_PREFIX['embedding']}{_hash_key(text)}"
            cached = await self._client.get(key)
            
            if cached:
                logger.debug("Embedding cache hit", text_hash=_hash_key(text))
                return json.loads(cached)
            
            return None
            
        except Exception as e:
            logger.warning("Embedding cache get failed", error=str(e))
            return None
    
    async def set_embedding(self, text: str, embedding: List[float]) -> bool:
        """Cache an embedding.
        
        Args:
            text: Text the embedding is for
            embedding: Embedding vector
            
        Returns:
            True if cached successfully
        """
        if not self._enabled:
            return False
        
        await self._ensure_connected()
        if not self._client:
            return False
        
        try:
            key = f"{CACHE_PREFIX['embedding']}{_hash_key(text)}"
            await self._client.setex(
                key,
                CACHE_TTL["embedding"],
                json.dumps(embedding),
            )
            return True
            
        except Exception as e:
            logger.warning("Embedding cache set failed", error=str(e))
            return False
    
    async def get_embeddings_batch(
        self,
        texts: List[str],
    ) -> Dict[int, List[float]]:
        """Get cached embeddings for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Dict mapping text index to embedding (only for cache hits)
        """
        if not self._enabled or not texts:
            return {}
        
        await self._ensure_connected()
        if not self._client:
            return {}
        
        try:
            keys = [f"{CACHE_PREFIX['embedding']}{_hash_key(t)}" for t in texts]
            cached = await self._client.mget(keys)
            
            results = {}
            for i, value in enumerate(cached):
                if value:
                    results[i] = json.loads(value)
            
            if results:
                logger.debug(
                    "Embedding batch cache hits",
                    hits=len(results),
                    total=len(texts),
                )
            
            return results
            
        except Exception as e:
            logger.warning("Embedding batch cache get failed", error=str(e))
            return {}
    
    async def set_embeddings_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]],
    ) -> bool:
        """Cache multiple embeddings.
        
        Args:
            texts: List of texts
            embeddings: List of embedding vectors
            
        Returns:
            True if cached successfully
        """
        if not self._enabled or not texts:
            return False
        
        await self._ensure_connected()
        if not self._client:
            return False
        
        try:
            pipe = self._client.pipeline()
            
            for text, embedding in zip(texts, embeddings):
                key = f"{CACHE_PREFIX['embedding']}{_hash_key(text)}"
                pipe.setex(key, CACHE_TTL["embedding"], json.dumps(embedding))
            
            await pipe.execute()
            return True
            
        except Exception as e:
            logger.warning("Embedding batch cache set failed", error=str(e))
            return False
    
    # =========================================================================
    # Classification Cache
    # =========================================================================
    
    async def get_classification(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached sector classification.
        
        Args:
            text: Text to classify
            
        Returns:
            Cached classification result or None
        """
        if not self._enabled:
            return None
        
        await self._ensure_connected()
        if not self._client:
            return None
        
        try:
            key = f"{CACHE_PREFIX['classification']}{_hash_key(text)}"
            cached = await self._client.get(key)
            
            if cached:
                logger.debug("Classification cache hit")
                return json.loads(cached)
            
            return None
            
        except Exception as e:
            logger.warning("Classification cache get failed", error=str(e))
            return None
    
    async def set_classification(
        self,
        text: str,
        classification: Dict[str, Any],
    ) -> bool:
        """Cache a classification result.
        
        Args:
            text: Text that was classified
            classification: Classification result
            
        Returns:
            True if cached successfully
        """
        if not self._enabled:
            return False
        
        await self._ensure_connected()
        if not self._client:
            return False
        
        try:
            key = f"{CACHE_PREFIX['classification']}{_hash_key(text)}"
            await self._client.setex(
                key,
                CACHE_TTL["classification"],
                json.dumps(classification),
            )
            return True
            
        except Exception as e:
            logger.warning("Classification cache set failed", error=str(e))
            return False
    
    # =========================================================================
    # Graph Neighbors Cache
    # =========================================================================
    
    async def get_graph_neighbors(
        self,
        memory_id: str,
        rel_types: List[str],
        hops: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached graph neighbors.
        
        Args:
            memory_id: Memory ID
            rel_types: Relationship types
            hops: Number of hops
            
        Returns:
            Cached neighbors or None
        """
        if not self._enabled:
            return None
        
        await self._ensure_connected()
        if not self._client:
            return None
        
        try:
            cache_key = f"{memory_id}:{','.join(sorted(rel_types))}:{hops}"
            key = f"{CACHE_PREFIX['graph_neighbors']}{_hash_key(cache_key)}"
            cached = await self._client.get(key)
            
            if cached:
                logger.debug("Graph neighbors cache hit", memory_id=memory_id)
                return json.loads(cached)
            
            return None
            
        except Exception as e:
            logger.warning("Graph cache get failed", error=str(e))
            return None
    
    async def set_graph_neighbors(
        self,
        memory_id: str,
        rel_types: List[str],
        hops: int,
        neighbors: List[Dict[str, Any]],
    ) -> bool:
        """Cache graph neighbors.
        
        Args:
            memory_id: Memory ID
            rel_types: Relationship types
            hops: Number of hops
            neighbors: Neighbor results
            
        Returns:
            True if cached successfully
        """
        if not self._enabled:
            return False
        
        await self._ensure_connected()
        if not self._client:
            return False
        
        try:
            cache_key = f"{memory_id}:{','.join(sorted(rel_types))}:{hops}"
            key = f"{CACHE_PREFIX['graph_neighbors']}{_hash_key(cache_key)}"
            await self._client.setex(
                key,
                CACHE_TTL["graph_neighbors"],
                json.dumps(neighbors),
            )
            return True
            
        except Exception as e:
            logger.warning("Graph cache set failed", error=str(e))
            return False
    
    # =========================================================================
    # Query Results Cache
    # =========================================================================
    
    async def get_query_results(
        self,
        query: str,
        tenant_id: str,
        sectors: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Get cached query results.
        
        Args:
            query: Query text
            tenant_id: Tenant ID
            sectors: Sectors searched
            
        Returns:
            Cached results or None
        """
        if not self._enabled:
            return None
        
        await self._ensure_connected()
        if not self._client:
            return None
        
        try:
            cache_key = f"{query}:{tenant_id}:{','.join(sorted(sectors))}"
            key = f"{CACHE_PREFIX['query_results']}{_hash_key(cache_key)}"
            cached = await self._client.get(key)
            
            if cached:
                logger.debug("Query results cache hit")
                return json.loads(cached)
            
            return None
            
        except Exception as e:
            logger.warning("Query cache get failed", error=str(e))
            return None
    
    async def set_query_results(
        self,
        query: str,
        tenant_id: str,
        sectors: List[str],
        results: Dict[str, Any],
    ) -> bool:
        """Cache query results.
        
        Args:
            query: Query text
            tenant_id: Tenant ID
            sectors: Sectors searched
            results: Query results
            
        Returns:
            True if cached successfully
        """
        if not self._enabled:
            return False
        
        await self._ensure_connected()
        if not self._client:
            return False
        
        try:
            cache_key = f"{query}:{tenant_id}:{','.join(sorted(sectors))}"
            key = f"{CACHE_PREFIX['query_results']}{_hash_key(cache_key)}"
            await self._client.setex(
                key,
                CACHE_TTL["query_results"],
                json.dumps(results),
            )
            return True
            
        except Exception as e:
            logger.warning("Query cache set failed", error=str(e))
            return False
    
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    async def invalidate_memory(self, memory_id: str):
        """Invalidate all caches related to a memory.
        
        Called when a memory is updated or deleted.
        
        Args:
            memory_id: Memory ID to invalidate
        """
        if not self._enabled:
            return
        
        await self._ensure_connected()
        if not self._client:
            return
        
        try:
            # Find and delete graph neighbor caches for this memory
            pattern = f"{CACHE_PREFIX['graph_neighbors']}*"
            async for key in self._client.scan_iter(match=pattern):
                # This is conservative - invalidates all graph caches
                # A more sophisticated approach would track which caches
                # contain which memory IDs
                await self._client.delete(key)
            
            logger.debug("Cache invalidated for memory", memory_id=memory_id)
            
        except Exception as e:
            logger.warning("Cache invalidation failed", error=str(e))
    
    async def clear_all(self):
        """Clear all performance caches."""
        if not self._enabled:
            return
        
        await self._ensure_connected()
        if not self._client:
            return
        
        try:
            for prefix in CACHE_PREFIX.values():
                async for key in self._client.scan_iter(match=f"{prefix}*"):
                    await self._client.delete(key)
            
            logger.info("All performance caches cleared")
            
        except Exception as e:
            logger.warning("Cache clear failed", error=str(e))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        if not self._enabled:
            return {"enabled": False}
        
        await self._ensure_connected()
        if not self._client:
            return {"enabled": False, "connected": False}
        
        try:
            stats = {
                "enabled": True,
                "connected": True,
            }
            
            for cache_type, prefix in CACHE_PREFIX.items():
                count = 0
                async for _ in self._client.scan_iter(match=f"{prefix}*"):
                    count += 1
                stats[f"{cache_type}_count"] = count
            
            return stats
            
        except Exception as e:
            return {"enabled": True, "connected": False, "error": str(e)}


# Global singleton
_perf_cache: Optional[PerformanceCache] = None


def get_perf_cache() -> PerformanceCache:
    """Get the global performance cache instance."""
    global _perf_cache
    if _perf_cache is None:
        _perf_cache = PerformanceCache()
    return _perf_cache



