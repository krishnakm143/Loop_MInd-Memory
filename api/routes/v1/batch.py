"""Batch Operations API Endpoints.

Provides bulk operations for efficient data processing:
- POST /v1/memories/batch - Create multiple memories
- POST /v1/query/batch - Execute multiple queries
- DELETE /v1/memories/batch - Bulk delete memories
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from core.db.session import get_async_session
from core.models.memory import Memory
from api.dependencies import get_current_user
from services.preprocessor import preprocess_text
from services.antiviral import scan_content
from services.sector_classifier import classify_sector
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/v1/batch", tags=["batch"])


# Request/Response Models
class BatchMemoryItem(BaseModel):
    """Single memory item for batch creation."""
    content: str
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class BatchMemoryRequest(BaseModel):
    """Request for batch memory creation."""
    memories: List[BatchMemoryItem] = Field(..., min_length=1, max_length=100)
    skip_antiviral: bool = False
    async_processing: bool = True


class BatchMemoryResultItem(BaseModel):
    """Result for a single memory in batch."""
    index: int
    memory_id: Optional[str] = None
    status: str  # created, failed, quarantined
    error: Optional[str] = None


class BatchMemoryResponse(BaseModel):
    """Response for batch memory creation."""
    batch_id: str
    total: int
    created: int
    failed: int
    quarantined: int
    results: List[BatchMemoryResultItem]
    processing_time_ms: int


class BatchQueryItem(BaseModel):
    """Single query item for batch execution."""
    query_text: str
    limit: int = 10
    filters: Dict[str, Any] = {}
    sectors: List[str] = []


class BatchQueryRequest(BaseModel):
    """Request for batch query execution."""
    queries: List[BatchQueryItem] = Field(..., min_length=1, max_length=50)


class BatchQueryResultItem(BaseModel):
    """Result for a single query in batch."""
    index: int
    query_id: str
    results: List[Dict[str, Any]]
    total: int
    latency_ms: int


class BatchQueryResponse(BaseModel):
    """Response for batch query execution."""
    batch_id: str
    total_queries: int
    results: List[BatchQueryResultItem]
    total_latency_ms: int


class BatchDeleteRequest(BaseModel):
    """Request for batch memory deletion."""
    memory_ids: List[str] = Field(..., min_length=1, max_length=100)
    hard_delete: bool = False  # Soft delete by default


class BatchDeleteResponse(BaseModel):
    """Response for batch deletion."""
    batch_id: str
    requested: int
    deleted: int
    failed: int
    errors: List[Dict[str, str]]


@router.post("/memories", response_model=BatchMemoryResponse)
async def batch_create_memories(
    request: BatchMemoryRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> BatchMemoryResponse:
    """Create multiple memories in a single request.
    
    Args:
        request: Batch memory creation request
        
    Returns:
        Results for each memory in the batch
    """
    import time
    start_time = time.time()
    
    batch_id = str(uuid.uuid4())
    tenant_id = user.get("tenant_id", "default")
    user_id = user.get("user_id", "unknown")
    
    results: List[BatchMemoryResultItem] = []
    created = 0
    failed = 0
    quarantined = 0
    
    for index, item in enumerate(request.memories):
        try:
            # Preprocess
            preprocessed = await preprocess_text(item.content)
            
            # ANTIVIRAL scan (unless skipped)
            trust_score = 1.0
            memory_status = "active"
            
            if not request.skip_antiviral:
                antiviral_result = await scan_content(preprocessed["normalized_text"])
                if antiviral_result["is_threat"]:
                    trust_score = antiviral_result["trust_score"]
                    if antiviral_result["threat_score"] > 0.85:
                        memory_status = "quarantined"
                        quarantined += 1
                        results.append(BatchMemoryResultItem(
                            index=index,
                            status="quarantined",
                            error=f"ANTIVIRAL: {antiviral_result['threat_type']}",
                        ))
                        continue
            
            # Classify sector
            sector_result = await classify_sector(preprocessed["normalized_text"])
            
            # Create memory
            content_data = preprocessed["normalized_text"]
            if isinstance(item.content, dict):
                content_data = item.content
                
            # Merge tags
            all_tags = list(item.tags) + preprocessed.get("extracted_tags", [])
            unique_tags = list(dict.fromkeys(all_tags))
            
            memory = Memory(
                tenant_id=tenant_id,
                user_id=user_id,
                type=sector_result["sector"],
                content=content_data,
                tags=unique_tags,
                status=memory_status,
                importance=0.5,
                trust_score=trust_score,
                version=1,
                provenance=[{
                    "source": "batch_api",
                    "batch_id": batch_id,
                    "index": index,
                    "timestamp": datetime.utcnow().isoformat(),
                }],
            )
            
            session.add(memory)
            await session.flush()
            
            results.append(BatchMemoryResultItem(
                index=index,
                memory_id=str(memory.memory_id),
                status="created",
            ))
            created += 1
            
        except Exception as e:
            logger.error(
                "Batch memory creation failed",
                batch_id=batch_id,
                index=index,
                error=str(e),
            )
            results.append(BatchMemoryResultItem(
                index=index,
                status="failed",
                error=str(e)[:100],
            ))
            failed += 1
    
    await session.commit()
    
    processing_time = int((time.time() - start_time) * 1000)
    
    logger.info(
        "Batch memory creation completed",
        batch_id=batch_id,
        total=len(request.memories),
        created=created,
        failed=failed,
        quarantined=quarantined,
        processing_time_ms=processing_time,
    )
    
    return BatchMemoryResponse(
        batch_id=batch_id,
        total=len(request.memories),
        created=created,
        failed=failed,
        quarantined=quarantined,
        results=results,
        processing_time_ms=processing_time,
    )


@router.post("/queries", response_model=BatchQueryResponse)
async def batch_execute_queries(
    request: BatchQueryRequest,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> BatchQueryResponse:
    """Execute multiple queries in a single request.
    
    Args:
        request: Batch query request
        
    Returns:
        Results for each query
    """
    import time
    start_time = time.time()
    
    batch_id = str(uuid.uuid4())
    tenant_id = user.get("tenant_id", "default")
    
    results: List[BatchQueryResultItem] = []
    
    for index, query_item in enumerate(request.queries):
        query_start = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # Build query
            stmt = select(Memory).where(
                Memory.tenant_id == tenant_id,
                Memory.status == "active",
            )
            
            # Apply sector filter
            if query_item.sectors:
                stmt = stmt.where(Memory.type.in_(query_item.sectors))
            
            # Limit results
            stmt = stmt.limit(query_item.limit)
            
            result = await session.execute(stmt)
            memories = result.scalars().all()
            
            # Format results
            memory_results = [
                {
                    "memory_id": str(m.memory_id),
                    "content": m.content[:200] if isinstance(m.content, str) else str(m.content)[:200],
                    "type": m.type,
                    "importance": m.importance,
                    "trust_score": m.trust_score,
                }
                for m in memories
            ]
            
            query_latency = int((time.time() - query_start) * 1000)
            
            results.append(BatchQueryResultItem(
                index=index,
                query_id=query_id,
                results=memory_results,
                total=len(memory_results),
                latency_ms=query_latency,
            ))
            
        except Exception as e:
            logger.error(
                "Batch query failed",
                batch_id=batch_id,
                index=index,
                error=str(e),
            )
            results.append(BatchQueryResultItem(
                index=index,
                query_id=query_id,
                results=[],
                total=0,
                latency_ms=0,
            ))
    
    total_latency = int((time.time() - start_time) * 1000)
    
    logger.info(
        "Batch query completed",
        batch_id=batch_id,
        total_queries=len(request.queries),
        total_latency_ms=total_latency,
    )
    
    return BatchQueryResponse(
        batch_id=batch_id,
        total_queries=len(request.queries),
        results=results,
        total_latency_ms=total_latency,
    )


@router.delete("/memories", response_model=BatchDeleteResponse)
async def batch_delete_memories(
    request: BatchDeleteRequest,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> BatchDeleteResponse:
    """Delete multiple memories in a single request.
    
    Args:
        request: Batch delete request
        
    Returns:
        Deletion results
    """
    batch_id = str(uuid.uuid4())
    tenant_id = user.get("tenant_id", "default")
    
    deleted = 0
    failed = 0
    errors: List[Dict[str, str]] = []
    
    for memory_id in request.memory_ids:
        try:
            # Verify ownership
            stmt = select(Memory).where(
                Memory.memory_id == memory_id,
                Memory.tenant_id == tenant_id,
            )
            result = await session.execute(stmt)
            memory = result.scalar_one_or_none()
            
            if not memory:
                errors.append({
                    "memory_id": memory_id,
                    "error": "Memory not found or access denied",
                })
                failed += 1
                continue
            
            if request.hard_delete:
                await session.delete(memory)
            else:
                memory.status = "deleted"
                
            deleted += 1
            
        except Exception as e:
            errors.append({
                "memory_id": memory_id,
                "error": str(e)[:100],
            })
            failed += 1
    
    await session.commit()
    
    logger.info(
        "Batch delete completed",
        batch_id=batch_id,
        requested=len(request.memory_ids),
        deleted=deleted,
        failed=failed,
    )
    
    return BatchDeleteResponse(
        batch_id=batch_id,
        requested=len(request.memory_ids),
        deleted=deleted,
        failed=failed,
        errors=errors,
    )

