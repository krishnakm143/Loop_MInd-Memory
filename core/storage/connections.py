"""
Unified Connection Pool Manager.

Manages async connection pools for all databases:
- PostgreSQL (SQLAlchemy async)
- Redis (redis-py async)
- Neo4j (async or sync driver)
- Qdrant (HTTP client)

Provides:
- Warm connections on startup
- Health-check keepalives
- Connection pooling optimization
- Graceful shutdown

Performance target: <100ms query latency via connection reuse.
"""

import asyncio
import structlog
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

from core.storage.cache import CacheStore
from core.storage.graph import GraphStore
from core.storage.vector import VectorStore
from core.db.session import engine, AsyncSessionLocal

logger = structlog.get_logger(__name__)


@dataclass
class ConnectionStats:
    """Track connection pool statistics."""
    
    postgresql_connections: int = 0
    redis_connections: int = 0
    neo4j_connections: int = 0
    qdrant_connections: int = 0
    
    last_health_check: Optional[datetime] = None
    health_check_failures: Dict[str, int] = field(default_factory=dict)
    
    postgresql_healthy: bool = False
    redis_healthy: bool = False
    neo4j_healthy: bool = False
    qdrant_healthy: bool = False


class ConnectionPoolManager:
    """
    Centralized manager for all database connections.
    
    Provides connection pooling, warm-up, and health monitoring
    for optimal query performance.
    """
    
    _instance: Optional["ConnectionPoolManager"] = None
    
    def __new__(cls):
        """Singleton pattern for global connection manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize connection stores."""
        if self._initialized:
            return
            
        self.cache: Optional[CacheStore] = None
        self.graph: Optional[GraphStore] = None
        self.vector: Optional[VectorStore] = None
        
        self.stats = ConnectionStats()
        self._health_check_task: Optional[asyncio.Task] = None
        self._initialized = True
        
        logger.info("ConnectionPoolManager initialized")
    
    async def warmup(self) -> Dict[str, bool]:
        """
        Warm up all connections on startup.
        
        Pre-establishes connections to reduce first-request latency.
        
        Returns:
            Dict mapping service name to connection success
        """
        logger.info("Warming up database connections...")
        results = {}
        
        # PostgreSQL - already pooled via SQLAlchemy
        try:
            async with AsyncSessionLocal() as session:
                await session.execute("SELECT 1")
            results["postgresql"] = True
            self.stats.postgresql_healthy = True
            logger.info("PostgreSQL connection warmed up")
        except Exception as e:
            results["postgresql"] = False
            self.stats.postgresql_healthy = False
            logger.error("PostgreSQL warmup failed", error=str(e))
        
        # Redis
        try:
            self.cache = CacheStore()
            await self.cache.connect()
            if await self.cache.health_check():
                results["redis"] = True
                self.stats.redis_healthy = True
                logger.info("Redis connection warmed up")
            else:
                results["redis"] = False
                self.stats.redis_healthy = False
        except Exception as e:
            results["redis"] = False
            self.stats.redis_healthy = False
            logger.error("Redis warmup failed", error=str(e))
        
        # Neo4j
        try:
            self.graph = GraphStore()
            if await self.graph.health_check():
                results["neo4j"] = True
                self.stats.neo4j_healthy = True
                logger.info("Neo4j connection warmed up")
            else:
                results["neo4j"] = False
                self.stats.neo4j_healthy = False
        except Exception as e:
            results["neo4j"] = False
            self.stats.neo4j_healthy = False
            logger.error("Neo4j warmup failed", error=str(e))
        
        # Qdrant
        try:
            self.vector = VectorStore()
            # Simple health check - list collections
            collections = self.vector.client.get_collections()
            results["qdrant"] = True
            self.stats.qdrant_healthy = True
            logger.info(
                "Qdrant connection warmed up",
                collections=len(collections.collections)
            )
        except Exception as e:
            results["qdrant"] = False
            self.stats.qdrant_healthy = False
            logger.error("Qdrant warmup failed", error=str(e))
        
        self.stats.last_health_check = datetime.utcnow()
        
        healthy_count = sum(1 for v in results.values() if v)
        logger.info(
            "Connection warmup complete",
            healthy=healthy_count,
            total=len(results),
            results=results,
        )
        
        return results
    
    async def start_health_checks(self, interval_seconds: int = 30):
        """
        Start background health check task.
        
        Periodically checks all connections and reconnects if needed.
        
        Args:
            interval_seconds: Time between health checks
        """
        async def _health_check_loop():
            while True:
                await asyncio.sleep(interval_seconds)
                await self._run_health_checks()
        
        self._health_check_task = asyncio.create_task(_health_check_loop())
        logger.info(
            "Health check task started",
            interval_seconds=interval_seconds,
        )
    
    async def stop_health_checks(self):
        """Stop the background health check task."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Health check task stopped")
    
    async def _run_health_checks(self):
        """Run health checks on all connections."""
        logger.debug("Running health checks...")
        
        # PostgreSQL
        try:
            async with AsyncSessionLocal() as session:
                await session.execute("SELECT 1")
            self.stats.postgresql_healthy = True
        except Exception as e:
            self.stats.postgresql_healthy = False
            self.stats.health_check_failures["postgresql"] = (
                self.stats.health_check_failures.get("postgresql", 0) + 1
            )
            logger.warning("PostgreSQL health check failed", error=str(e))
        
        # Redis
        if self.cache:
            try:
                if await self.cache.health_check():
                    self.stats.redis_healthy = True
                else:
                    self.stats.redis_healthy = False
                    # Try to reconnect
                    await self.cache.disconnect()
                    await self.cache.connect()
            except Exception as e:
                self.stats.redis_healthy = False
                self.stats.health_check_failures["redis"] = (
                    self.stats.health_check_failures.get("redis", 0) + 1
                )
                logger.warning("Redis health check failed", error=str(e))
        
        # Neo4j
        if self.graph:
            try:
                if await self.graph.health_check():
                    self.stats.neo4j_healthy = True
                else:
                    self.stats.neo4j_healthy = False
            except Exception as e:
                self.stats.neo4j_healthy = False
                self.stats.health_check_failures["neo4j"] = (
                    self.stats.health_check_failures.get("neo4j", 0) + 1
                )
                logger.warning("Neo4j health check failed", error=str(e))
        
        # Qdrant
        if self.vector:
            try:
                self.vector.client.get_collections()
                self.stats.qdrant_healthy = True
            except Exception as e:
                self.stats.qdrant_healthy = False
                self.stats.health_check_failures["qdrant"] = (
                    self.stats.health_check_failures.get("qdrant", 0) + 1
                )
                logger.warning("Qdrant health check failed", error=str(e))
        
        self.stats.last_health_check = datetime.utcnow()
    
    def get_cache(self) -> CacheStore:
        """Get the Redis cache store."""
        if not self.cache:
            self.cache = CacheStore()
        return self.cache
    
    def get_graph(self) -> GraphStore:
        """Get the Neo4j graph store."""
        if not self.graph:
            self.graph = GraphStore()
        return self.graph
    
    def get_vector(self) -> VectorStore:
        """Get the Qdrant vector store."""
        if not self.vector:
            self.vector = VectorStore()
        return self.vector
    
    async def shutdown(self):
        """
        Gracefully shutdown all connections.
        
        Should be called during application shutdown.
        """
        logger.info("Shutting down connection pools...")
        
        await self.stop_health_checks()
        
        # Close Redis
        if self.cache:
            await self.cache.disconnect()
            logger.info("Redis connection closed")
        
        # Close Neo4j
        if self.graph:
            await self.graph.close()
            logger.info("Neo4j connection closed")
        
        # Qdrant client doesn't need explicit close
        
        # PostgreSQL engine close is handled separately
        await engine.dispose()
        logger.info("PostgreSQL connection pool closed")
        
        logger.info("All connections closed")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current connection pool status.
        
        Returns:
            Dict with health status and statistics
        """
        return {
            "postgresql": {
                "healthy": self.stats.postgresql_healthy,
                "failures": self.stats.health_check_failures.get("postgresql", 0),
            },
            "redis": {
                "healthy": self.stats.redis_healthy,
                "failures": self.stats.health_check_failures.get("redis", 0),
            },
            "neo4j": {
                "healthy": self.stats.neo4j_healthy,
                "failures": self.stats.health_check_failures.get("neo4j", 0),
            },
            "qdrant": {
                "healthy": self.stats.qdrant_healthy,
                "failures": self.stats.health_check_failures.get("qdrant", 0),
            },
            "last_health_check": (
                self.stats.last_health_check.isoformat()
                if self.stats.last_health_check else None
            ),
            "all_healthy": all([
                self.stats.postgresql_healthy,
                self.stats.redis_healthy,
                self.stats.neo4j_healthy,
                self.stats.qdrant_healthy,
            ]),
        }


# Global singleton instance
_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager() -> ConnectionPoolManager:
    """Get the global connection pool manager."""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager


async def warmup_connections() -> Dict[str, bool]:
    """Warm up all database connections (convenience function)."""
    manager = get_pool_manager()
    return await manager.warmup()


async def shutdown_connections():
    """Shutdown all connections (convenience function)."""
    manager = get_pool_manager()
    await manager.shutdown()



