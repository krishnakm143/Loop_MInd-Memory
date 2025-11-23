"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from core.db.session import get_async_session
from core.storage.vector import VectorStore
from core.storage.graph import GraphStore
from core.storage.cache import CacheStore

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "LoopMind_Memory-api"}


@router.get("/db")
async def health_db(session: AsyncSession = Depends(get_async_session)):
    """Check database connectivity."""
    try:
        result = await session.execute(text("SELECT 1"))
        result.scalar()
        return {"status": "healthy", "service": "postgres"}
    except Exception as e:
        return {"status": "unhealthy", "service": "postgres", "error": str(e)}


@router.get("/redis")
async def health_redis():
    """Check Redis connectivity."""
    cache = CacheStore()
    try:
        is_healthy = await cache.health_check()
        await cache.disconnect()
        if is_healthy:
            return {"status": "healthy", "service": "redis"}
        else:
            return {"status": "unhealthy", "service": "redis"}
    except Exception as e:
        return {"status": "unhealthy", "service": "redis", "error": str(e)}


@router.get("/qdrant")
async def health_qdrant():
    """Check Qdrant connectivity."""
    vector_store = VectorStore()
    try:
        is_healthy = await vector_store.health_check()
        if is_healthy:
            return {"status": "healthy", "service": "qdrant"}
        else:
            return {"status": "unhealthy", "service": "qdrant"}
    except Exception as e:
        return {"status": "unhealthy", "service": "qdrant", "error": str(e)}


@router.get("/neo4j")
async def health_neo4j():
    """Check Neo4j connectivity."""
    graph_store = GraphStore()
    try:
        is_healthy, error_msg = await graph_store.health_check()
        await graph_store.close()
        if is_healthy:
            return {"status": "healthy", "service": "neo4j"}
        else:
            return {"status": "unhealthy", "service": "neo4j", "error": error_msg or "Health check returned False"}
    except Exception as e:
        return {"status": "unhealthy", "service": "neo4j", "error": str(e)}