"""
Ultra-fast query endpoint using TurboRetriever.

Target latencies:
- Turbo mode: <10ms
- Fast mode: <50ms
- Balanced mode: <100ms
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException, status
from pydantic import BaseModel, Field
import time
import structlog

from api.dependencies import get_current_user
from services.turbo_retriever import get_turbo_retriever, TurboRetriever, RetrievalResult

logger = structlog.get_logger()
router = APIRouter(prefix="/fast", tags=["fast-query"])


class FastQueryRequest(BaseModel):
    """Fast query request model."""
    query: str = Field(..., min_length=1, max_length=10000)
    k: int = Field(default=10, ge=1, le=100)
    sectors: Optional[List[str]] = Field(default=None)
    min_salience: float = Field(default=0.0, ge=0.0, le=1.0)
    mode: str = Field(
        default="turbo",
        description="Query mode: turbo (<10ms), fast (<50ms), balanced (<100ms)"
    )


class FastQueryResult(BaseModel):
    """Single result from fast query."""
    memory_id: str
    content: str
    score: float
    sector: str
    similarity: float


class FastQueryResponse(BaseModel):
    """Fast query response model."""
    results: List[FastQueryResult]
    stats: Dict[str, Any]
    latency_ms: float


class BatchQueryRequest(BaseModel):
    """Batch query request model."""
    queries: List[str] = Field(..., min_items=1, max_items=100)
    k: int = Field(default=10, ge=1, le=100)


class BatchQueryResponse(BaseModel):
    """Batch query response model."""
    results: List[List[FastQueryResult]]
    total_latency_ms: float
    avg_latency_ms: float


class IndexMemoryRequest(BaseModel):
    """Request to index a memory for fast retrieval."""
    memory_id: str
    content: str
    sector: str = Field(default="semantic")
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    mode: str
    total_queries: int
    cache_hit_rate: float
    average_time_ms: float
    indexed_memories: int
    cache_stats: Dict[str, Any]


@router.post("/query", response_model=FastQueryResponse)
async def fast_query(
    request: FastQueryRequest,
    user: dict = Depends(get_current_user),
) -> FastQueryResponse:
    """
    Execute an ultra-fast memory query.
    
    Modes:
    - turbo: <10ms latency using synthetic embeddings
    - fast: <50ms latency with caching
    - balanced: <100ms with hybrid embeddings
    """
    start_time = time.time()
    tenant_id = user.get("tenant_id", "default")
    
    try:
        retriever = get_turbo_retriever(mode=request.mode)
        
        results, stats = await retriever.query(
            query_text=request.query,
            k=request.k,
            tenant_id=tenant_id,
            sectors=request.sectors,
            min_salience=request.min_salience,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "fast_query_executed",
            mode=request.mode,
            k=request.k,
            results_count=len(results),
            latency_ms=elapsed_ms,
            cache_hit=stats.cache_hit,
        )
        
        return FastQueryResponse(
            results=[
                FastQueryResult(
                    memory_id=r.memory_id,
                    content=r.content,
                    score=r.score,
                    sector=r.sector,
                    similarity=r.similarity,
                )
                for r in results
            ],
            stats={
                "mode": stats.mode,
                "cache_hit": stats.cache_hit,
                "embedding_time_ms": stats.embedding_time_ms,
                "search_time_ms": stats.search_time_ms,
                "scoring_time_ms": stats.scoring_time_ms,
            },
            latency_ms=elapsed_ms,
        )
        
    except Exception as e:
        logger.error("fast_query_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchQueryResponse)
async def batch_query(
    request: BatchQueryRequest,
    user: dict = Depends(get_current_user),
) -> BatchQueryResponse:
    """
    Execute multiple queries in parallel for maximum throughput.
    """
    start_time = time.time()
    tenant_id = user.get("tenant_id", "default")
    
    try:
        retriever = get_turbo_retriever(mode="turbo")
        
        batch_results = await retriever.batch_query(
            queries=request.queries,
            k=request.k,
            tenant_id=tenant_id,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        avg_ms = elapsed_ms / len(request.queries)
        
        logger.info(
            "batch_query_executed",
            queries_count=len(request.queries),
            total_latency_ms=elapsed_ms,
            avg_latency_ms=avg_ms,
        )
        
        return BatchQueryResponse(
            results=[
                [
                    FastQueryResult(
                        memory_id=r.memory_id,
                        content=r.content,
                        score=r.score,
                        sector=r.sector,
                        similarity=r.similarity,
                    )
                    for r in results
                ]
                for results, _ in batch_results
            ],
            total_latency_ms=elapsed_ms,
            avg_latency_ms=avg_ms,
        )
        
    except Exception as e:
        logger.error("batch_query_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch query failed: {str(e)}"
        )


@router.post("/index")
async def index_memory(
    request: IndexMemoryRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Index a memory for ultra-fast retrieval.
    
    This pre-computes synthetic embeddings and stores them in memory.
    """
    try:
        retriever = get_turbo_retriever()
        
        await retriever.index_memory(
            memory_id=request.memory_id,
            content=request.content,
            sector=request.sector,
            salience=request.salience,
            metadata=request.metadata,
        )
        
        logger.info(
            "memory_indexed",
            memory_id=request.memory_id,
            sector=request.sector,
        )
        
        return {
            "status": "indexed",
            "memory_id": request.memory_id,
            "sector": request.sector,
        }
        
    except Exception as e:
        logger.error("index_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    user: dict = Depends(get_current_user),
) -> CacheStatsResponse:
    """Get cache and retriever statistics."""
    try:
        retriever = get_turbo_retriever()
        stats = retriever.stats()
        
        return CacheStatsResponse(
            mode=stats["mode"],
            total_queries=stats["total_queries"],
            cache_hit_rate=stats["cache_hit_rate"],
            average_time_ms=stats["average_time_ms"],
            indexed_memories=stats["indexed_memories"],
            cache_stats=stats["cache_stats"],
        )
        
    except Exception as e:
        logger.error("stats_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.post("/warmup")
async def warmup_cache(
    user: dict = Depends(get_current_user),
) -> Dict[str, str]:
    """Warmup the cache with common patterns."""
    try:
        retriever = get_turbo_retriever()
        
        # Index some test memories for warmup
        test_memories = [
            ("warmup_1", "Test memory for cache warmup", "semantic"),
            ("warmup_2", "Another test memory for retrieval", "episodic"),
            ("warmup_3", "How to configure the system", "procedural"),
        ]
        
        for mem_id, content, sector in test_memories:
            await retriever.index_memory(mem_id, content, sector)
        
        logger.info("cache_warmup_complete")
        
        return {"status": "warmup_complete", "indexed": len(test_memories)}
        
    except Exception as e:
        logger.error("warmup_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Warmup failed: {str(e)}"
        )



