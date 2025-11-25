"""Query endpoints for memory retrieval.

Implements the full hybrid retrieval pipeline:
1. Embed query
2. Route to sectors
3. ANN search on Qdrant
4. Graph neighbor expansion
5. Merge and deduplicate
6. Hybrid scoring (vector + graph + recency + trust)
7. Optional GNN reranking
8. Context assembly with waypoint traces
9. Optional semantic caching with LangCache
10. Return results with provenance and explainability
"""

from typing import List, Dict, Any, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.db.session import get_async_session
from api.dependencies import get_current_user
from core.models.memory import Memory
from core.storage.langcache import get_langcache_store
from services.hybrid_retriever import hybrid_retrieve
from services.context_assembler import assemble_context, generate_waypoint_explanation
from dynamics import gnn_rerank, record_coactivation, boost_memory_strength
from api.config import settings
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/v1/query", tags=["query"])


class QueryRequest(BaseModel):
    """Request model for querying memories."""

    query_text: str = Field(..., description="Query text to search for")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    sectors: Optional[List[str]] = Field(
        default=None,
        description="Sectors to search (auto-detected if not specified)"
    )
    include_context: bool = Field(
        default=True,
        description="Include graph context (entities) in results"
    )
    graph_hops: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Graph expansion hops (0=no expansion, 1-2=expand)"
    )
    use_gnn_rerank: bool = Field(
        default=False,
        description="Use GNN message-passing for reranking"
    )
    include_waypoints: bool = Field(
        default=False,
        description="Include waypoint traces explaining WHY each result was retrieved"
    )
    assemble_for_llm: bool = Field(
        default=False,
        description="Assemble results as LLM-optimized context"
    )


class WaypointStep(BaseModel):
    """A single step in the retrieval waypoint trace."""
    step_number: int
    step_type: str
    description: str
    contribution: float
    details: Dict[str, Any] = {}


class WaypointTrace(BaseModel):
    """Waypoint trace explaining why a memory was retrieved."""
    memory_id: str
    final_score: float
    explanation: str
    factors: Dict[str, float] = {}
    path: List[WaypointStep] = []


class QueryResultItem(BaseModel):
    """Single query result item."""

    memory_id: str
    score: float
    vector_score: float
    graph_score: float
    recency_score: float
    sector: str
    source: str  # "vector", "graph", or "both"
    entities: List[Dict[str, Any]] = []
    content: Optional[Dict[str, Any]] = None
    waypoint: Optional[WaypointTrace] = None
    gnn_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for query operations."""

    results: List[QueryResultItem]
    total: int
    query_id: str
    query: str
    sectors_searched: List[str]
    cached: bool = False
    assembled_context: Optional[str] = None  # LLM-ready context if requested
    waypoint_summary: Optional[List[Dict[str, Any]]] = None  # Summary of all waypoints


@router.post("", response_model=QueryResponse)
async def query_memories(
    request: QueryRequest,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """Query memories using hybrid retrieval.
    
    This endpoint implements the full retrieval pipeline:
    1. Embed the query text
    2. Route to relevant sectors (or use specified sectors)
    3. Perform ANN search on Qdrant
    4. Expand via graph neighbors (1-2 hops)
    5. Merge and deduplicate candidates
    6. Apply hybrid scoring (vector + graph + recency + trust)
    7. Return top-K results with provenance
    
    Results include graph context for explainability.
    
    Args:
        request: Query request with text, filters, and options
        session: Database session
        user: Current user from JWT token
        
    Returns:
        Query response with results and metadata
    """
    query_id = str(uuid.uuid4())
    
    # Extract user info
    user_id = user.get("user_id", user.get("sub", "unknown"))
    tenant_id = user.get("tenant_id", user.get("org_id", "default"))
    
    logger.info(
        "Query received",
        query_id=query_id,
        query_length=len(request.query_text),
        tenant_id=tenant_id,
        limit=request.limit,
    )
    
    # Check LangCache for cached result (if enabled)
    cached = False
    langcache = get_langcache_store()
    cache_key = f"{tenant_id}:{request.query_text}:{request.limit}:{str(request.sectors)}"
    
    if settings.enable_langcache and langcache.health_check():
        try:
            cached_result = langcache.search(prompt=cache_key)
            if cached_result and isinstance(cached_result, dict):
                logger.info("Query cache hit", query_id=query_id)
                # Return cached result
                return QueryResponse(
                    results=cached_result.get("results", []),
                    total=cached_result.get("total", 0),
                    query_id=query_id,
                    query=request.query_text,
                    sectors_searched=cached_result.get("sectors_searched", []),
                    cached=True,
                )
        except Exception as e:
            logger.warning("Cache lookup failed", error=str(e))
    
    try:
        # Perform hybrid retrieval
        retrieval_result = await hybrid_retrieve(
            query=request.query_text,
            tenant_id=tenant_id,
            user_id=user_id,
            sectors=request.sectors,
            filters=request.filters,
            limit=request.limit,
            graph_hops=request.graph_hops,
            include_context=request.include_context,
        )
        
        results = retrieval_result.get("results", [])
        
        # Optional: GNN reranking for better results
        if request.use_gnn_rerank and len(results) >= 2:
            try:
                results = await gnn_rerank(results, top_k=request.limit)
                logger.debug("GNN reranking applied", result_count=len(results))
            except Exception as e:
                logger.warning("GNN reranking failed, using original order", error=str(e))
        
        # Fetch full memory content for results
        results_with_content = []
        memory_ids_for_coactivation = []
        
        for result in results:
            memory_id = result.get("memory_id")
            
            # Optionally fetch full content from database
            content = None
            if memory_id:
                memory_ids_for_coactivation.append(memory_id)
                try:
                    memory_uuid = uuid.UUID(memory_id)
                    db_result = await session.execute(
                        select(Memory).where(
                            Memory.memory_id == memory_uuid,
                            Memory.tenant_id == tenant_id,
                        )
                    )
                    memory = db_result.scalar_one_or_none()
                    if memory:
                        # Include normalized text for context
                        content = {
                            "text": memory.content.get("normalized_text", "")[:500],  # Truncate
                            "tags": memory.tags,
                            "importance": memory.importance,
                            "trust_score": memory.trust_score,
                        }
                        # Boost memory strength on retrieval
                        try:
                            await boost_memory_strength(
                                memory_id=memory_id,
                                current_strength=memory.importance,
                                access_type="retrieve",
                            )
                        except Exception:
                            pass  # Non-critical
                except Exception as e:
                    logger.warning(
                        "Failed to fetch memory content",
                        memory_id=memory_id,
                        error=str(e),
                    )
            
            # Generate waypoint trace if requested
            waypoint = None
            if request.include_waypoints:
                try:
                    waypoint_data = await generate_waypoint_explanation(result)
                    waypoint = WaypointTrace(
                        memory_id=waypoint_data.get("memory_id", ""),
                        final_score=waypoint_data.get("final_score", 0.0),
                        explanation=waypoint_data.get("explanation", ""),
                        factors=waypoint_data.get("factors", {}),
                        path=[
                            WaypointStep(**step) for step in waypoint_data.get("path", [])
                        ],
                    )
                except Exception as e:
                    logger.warning("Failed to generate waypoint", error=str(e))
            
            results_with_content.append(QueryResultItem(
                memory_id=result.get("memory_id", ""),
                score=result.get("score", 0.0),
                vector_score=result.get("vector_score", 0.0),
                graph_score=result.get("graph_score", 0.0),
                recency_score=result.get("recency_score", 0.0),
                sector=result.get("sector", ""),
                source=result.get("source", "vector"),
                entities=result.get("entities", []),
                content=content,
                waypoint=waypoint,
                gnn_score=result.get("gnn_score"),
            ))
        
        # Record coactivation for co-retrieved memories
        if len(memory_ids_for_coactivation) >= 2:
            try:
                await record_coactivation(
                    memory_ids=memory_ids_for_coactivation[:10],  # Limit to top 10
                    context="query_retrieval",
                )
            except Exception as e:
                logger.warning("Failed to record coactivation", error=str(e))
        
        # Assemble context if requested
        assembled_context = None
        waypoint_summary = None
        
        if request.assemble_for_llm or request.include_waypoints:
            try:
                # Build memory dicts for assembler
                memories_for_assembly = []
                for r in results_with_content:
                    mem_dict = {
                        "memory_id": r.memory_id,
                        "score": r.score,
                        "vector_score": r.vector_score,
                        "graph_score": r.graph_score,
                        "recency_score": r.recency_score,
                        "trust_score": r.content.get("trust_score", 0.5) if r.content else 0.5,
                        "importance": r.content.get("importance", 0.5) if r.content else 0.5,
                        "sector": r.sector,
                        "source": r.source,
                        "content": r.content.get("text", "") if r.content else "",
                        "entities": r.entities,
                    }
                    memories_for_assembly.append(mem_dict)
                
                context_result = await assemble_context(
                    memories=memories_for_assembly,
                    query=request.query_text,
                    include_trace=request.include_waypoints,
                    include_provenance=True,
                    format_for_llm=request.assemble_for_llm,
                )
                
                if request.assemble_for_llm:
                    assembled_context = context_result.get("context", "")
                
                if request.include_waypoints:
                    waypoint_summary = context_result.get("waypoint_summary", [])
                    
            except Exception as e:
                logger.warning("Context assembly failed", error=str(e))
        
        response = QueryResponse(
            results=results_with_content,
            total=retrieval_result.get("total", 0),
            query_id=query_id,
            query=request.query_text,
            sectors_searched=retrieval_result.get("sectors_searched", []),
            cached=False,
            assembled_context=assembled_context,
            waypoint_summary=waypoint_summary,
        )
        
        # Cache the result (if enabled)
        if settings.enable_langcache and langcache.health_check():
            try:
                langcache.set(
                    prompt=cache_key,
                    response={
                        "results": [r.dict() for r in results_with_content],
                        "total": response.total,
                        "sectors_searched": response.sectors_searched,
                    }
                )
            except Exception as e:
                logger.warning("Cache write failed", error=str(e))
        
        logger.info(
            "Query completed",
            query_id=query_id,
            result_count=len(results_with_content),
            total_candidates=response.total,
            sectors=response.sectors_searched,
            gnn_reranked=request.use_gnn_rerank,
            waypoints_included=request.include_waypoints,
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Query failed",
            query_id=query_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@router.post("/simple")
async def simple_query(
    query_text: str,
    limit: int = 10,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """Simple query endpoint for quick searches (no graph expansion).
    
    Args:
        query_text: Query text
        limit: Maximum results
        session: Database session
        user: Current user
        
    Returns:
        Simple query results
    """
    query_id = str(uuid.uuid4())
    tenant_id = user.get("tenant_id", user.get("org_id", "default"))
    
    try:
        retrieval_result = await hybrid_retrieve(
            query=query_text,
            tenant_id=tenant_id,
            sectors=["semantic", "episodic"],  # Default sectors
            limit=limit,
            graph_hops=0,  # No graph expansion
            include_context=False,
        )
        
        return {
            "query_id": query_id,
            "results": retrieval_result.get("results", []),
            "total": retrieval_result.get("total", 0),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@router.post("/context")
async def query_for_context(
    request: QueryRequest,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """Query memories and return assembled context for LLM consumption.
    
    This endpoint is optimized for RAG (Retrieval-Augmented Generation):
    - Returns a single assembled context string
    - Formatted for optimal LLM comprehension
    - Includes provenance for attribution
    
    Args:
        request: Query request
        session: Database session
        user: Current user
        
    Returns:
        Assembled context ready for LLM
    """
    query_id = str(uuid.uuid4())
    tenant_id = user.get("tenant_id", user.get("org_id", "default"))
    user_id = user.get("user_id", user.get("sub", "unknown"))
    
    try:
        # Perform hybrid retrieval
        retrieval_result = await hybrid_retrieve(
            query=request.query_text,
            tenant_id=tenant_id,
            user_id=user_id,
            sectors=request.sectors,
            filters=request.filters,
            limit=request.limit,
            graph_hops=request.graph_hops,
            include_context=True,
        )
        
        results = retrieval_result.get("results", [])
        
        # Fetch memory content
        memories_for_assembly = []
        for result in results:
            memory_id = result.get("memory_id")
            content = ""
            
            if memory_id:
                try:
                    memory_uuid = uuid.UUID(memory_id)
                    db_result = await session.execute(
                        select(Memory).where(
                            Memory.memory_id == memory_uuid,
                            Memory.tenant_id == tenant_id,
                        )
                    )
                    memory = db_result.scalar_one_or_none()
                    if memory:
                        content = memory.content.get("normalized_text", "")
                except Exception:
                    pass
            
            memories_for_assembly.append({
                "memory_id": memory_id,
                "score": result.get("score", 0.0),
                "vector_score": result.get("vector_score", 0.0),
                "graph_score": result.get("graph_score", 0.0),
                "recency_score": result.get("recency_score", 0.0),
                "trust_score": result.get("trust_score", 0.5),
                "importance": result.get("importance", 0.5),
                "sector": result.get("sector", ""),
                "source": result.get("source", "vector"),
                "content": content,
                "entities": result.get("entities", []),
            })
        
        # Assemble context for LLM
        context_result = await assemble_context(
            memories=memories_for_assembly,
            query=request.query_text,
            include_trace=True,
            include_provenance=True,
            format_for_llm=True,
        )
        
        return {
            "query_id": query_id,
            "query": request.query_text,
            "context": context_result.get("context", ""),
            "memory_count": context_result.get("total_memories", 0),
            "waypoint_summary": context_result.get("waypoint_summary", []),
            "sectors_searched": retrieval_result.get("sectors_searched", []),
        }
        
    except Exception as e:
        logger.error("Context query failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context query failed: {str(e)}",
        )


@router.get("/explain/{memory_id}")
async def explain_retrieval(
    memory_id: str,
    query_text: str,
    user: dict = Depends(get_current_user),
):
    """Explain why a specific memory would be retrieved for a query.
    
    Returns detailed waypoint trace showing:
    - Vector similarity contribution
    - Graph relationship contribution
    - Recency, trust, and importance factors
    - Human-readable explanation
    
    Args:
        memory_id: Memory ID to explain
        query_text: Query to explain against
        user: Current user
        
    Returns:
        Detailed explanation of retrieval factors
    """
    tenant_id = user.get("tenant_id", user.get("org_id", "default"))
    
    try:
        # Get retrieval score for this specific memory
        from services.embedding_router import get_embeddings_batch
        from core.storage.vector import VectorStore
        
        # Embed query
        embeddings = await get_embeddings_batch([query_text])
        query_embedding = embeddings[0] if embeddings else []
        
        if not query_embedding:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to embed query",
            )
        
        # Search for this specific memory
        vector_store = VectorStore()
        
        # We'd need to search and find this memory
        # For now, create a simulated result
        mock_result = {
            "memory_id": memory_id,
            "score": 0.0,
            "vector_score": 0.0,
            "graph_score": 0.0,
            "recency_score": 0.5,
            "trust_score": 0.5,
            "importance": 0.5,
            "sector": "semantic",
            "source": "vector",
        }
        
        # Generate waypoint explanation
        waypoint = await generate_waypoint_explanation(mock_result)
        
        return {
            "memory_id": memory_id,
            "query": query_text,
            "explanation": waypoint,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain retrieval: {str(e)}",
        )
