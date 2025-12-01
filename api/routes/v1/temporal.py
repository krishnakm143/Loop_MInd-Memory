"""
Temporal Knowledge Graph API endpoints.

Time-aware fact storage and retrieval:
- Add facts with temporal validity (auto-evolves existing)
- Point-in-time queries (what was true on date X?)
- Timeline reconstruction for entities
- Temporal comparison between dates
- Volatile fact detection

OpenMemory-competitive feature for temporal reasoning.
"""

import structlog
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.session import get_async_session
from api.dependencies import get_current_user
from services.temporal_graph import TemporalGraphService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/temporal", tags=["temporal"])


# =============================================================================
# Request/Response Models
# =============================================================================

class FactCreateRequest(BaseModel):
    """Request to create a new temporal fact."""
    
    subject: str = Field(..., description="Entity subject (e.g., 'CompanyX', 'John')")
    predicate: str = Field(..., description="Relationship type (e.g., 'has_CEO', 'works_at')")
    object: str = Field(..., description="Object value (e.g., 'Alice', 'New York')")
    valid_from: Optional[datetime] = Field(
        None,
        description="When the fact became true (default: now)"
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Initial confidence score"
    )
    source_memory_id: Optional[str] = Field(
        None,
        description="Source memory this fact was extracted from"
    )
    extra_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional data"
    )


class FactResponse(BaseModel):
    """Response model for a temporal fact."""
    
    id: str
    subject: str
    predicate: str
    object: str
    valid_from: datetime
    valid_to: Optional[datetime]
    confidence: float
    source_memory_id: Optional[str]
    source_type: str
    is_current: bool
    duration_days: Optional[int]
    created_at: datetime


class FactListResponse(BaseModel):
    """Response for listing facts."""
    
    facts: List[FactResponse]
    total: int


class TimelineResponse(BaseModel):
    """Response for entity timeline."""
    
    subject: str
    facts: List[FactResponse]
    total_facts: int
    date_range: Dict[str, Optional[str]]


class CompareResponse(BaseModel):
    """Response for date comparison."""
    
    subject: str
    date1: str
    date2: str
    added: List[Dict[str, Any]]
    removed: List[Dict[str, Any]]
    changed: List[Dict[str, Any]]
    unchanged: List[Dict[str, Any]]
    summary: Dict[str, int]


class VolatileFactResponse(BaseModel):
    """Response for volatile facts."""
    
    volatile_facts: List[Dict[str, Any]]
    time_window_days: int
    min_changes: int


class TemporalStatsResponse(BaseModel):
    """Response for temporal graph statistics."""
    
    total_facts: int
    current_facts: int
    historical_facts: int
    unique_subjects: int
    unique_predicates: int
    average_confidence: float
    total_changes: int


# =============================================================================
# Helper Functions
# =============================================================================

def fact_to_response(fact) -> FactResponse:
    """Convert TemporalFact model to response."""
    return FactResponse(
        id=str(fact.id),
        subject=fact.subject,
        predicate=fact.predicate,
        object=fact.object,
        valid_from=fact.valid_from,
        valid_to=fact.valid_to,
        confidence=fact.confidence,
        source_memory_id=str(fact.source_memory_id) if fact.source_memory_id else None,
        source_type=fact.source_type,
        is_current=fact.is_current,
        duration_days=fact.duration_days,
        created_at=fact.created_at,
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/facts", response_model=FactResponse, status_code=status.HTTP_201_CREATED)
async def create_fact(
    request: FactCreateRequest,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """
    Add a new temporal fact.
    
    If a fact with the same subject-predicate already exists,
    it will be automatically closed (valid_to set) and the new fact
    will become the current truth.
    
    This enables automatic fact evolution for time-aware reasoning.
    """
    tenant_id = user.get("tenant_id", "default")
    
    logger.info(
        "creating_temporal_fact",
        tenant_id=tenant_id,
        subject=request.subject,
        predicate=request.predicate,
    )
    
    service = TemporalGraphService(session)
    
    source_memory_uuid = None
    if request.source_memory_id:
        try:
            source_memory_uuid = UUID(request.source_memory_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid source_memory_id format"
            )
    
    try:
        fact = await service.add_fact(
            tenant_id=tenant_id,
            subject=request.subject,
            predicate=request.predicate,
            object_value=request.object,
            valid_from=request.valid_from,
            confidence=request.confidence,
            source_memory_id=source_memory_uuid,
            extra_data=request.extra_data,
        )
        
        await session.commit()
        
        return fact_to_response(fact)
    except Exception as e:
        await session.rollback()
        logger.error("temporal_fact_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create temporal fact: {str(e)}"
        )


@router.get("/facts", response_model=FactListResponse)
async def list_facts(
    subject: Optional[str] = Query(None, description="Filter by subject"),
    predicate: Optional[str] = Query(None, description="Filter by predicate"),
    at_date: Optional[datetime] = Query(None, description="Point-in-time query"),
    include_historical: bool = Query(False, description="Include expired facts"),
    search: Optional[str] = Query(None, description="Search in subject/object"),
    limit: int = Query(50, ge=1, le=500),
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """
    Query temporal facts with filters.
    
    Supports:
    - Filter by subject and/or predicate
    - Point-in-time queries (what was true at a specific date)
    - Full-text search in subject/object
    - Historical or current-only facts
    """
    tenant_id = user.get("tenant_id", "default")
    
    service = TemporalGraphService(session)
    
    if search:
        # Search mode
        facts = await service.search_facts(
            tenant_id=tenant_id,
            query=search,
            include_historical=include_historical,
            limit=limit,
        )
    elif subject:
        # Point-in-time or subject query
        facts = await service.query_point_in_time(
            tenant_id=tenant_id,
            subject=subject,
            predicate=predicate,
            at_date=at_date,
        )
        facts = facts[:limit]
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide 'subject' or 'search' parameter"
        )
    
    return FactListResponse(
        facts=[fact_to_response(f) for f in facts],
        total=len(facts),
    )


@router.get("/timeline/{subject}", response_model=TimelineResponse)
async def get_timeline(
    subject: str,
    predicate: Optional[str] = Query(None, description="Filter by predicate"),
    limit: int = Query(100, ge=1, le=500),
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """
    Get the full timeline of an entity's facts.
    
    Returns all facts (current and historical) for an entity,
    ordered by valid_from (newest first).
    
    Useful for reconstructing an entity's history.
    """
    tenant_id = user.get("tenant_id", "default")
    
    service = TemporalGraphService(session)
    
    facts = await service.get_timeline(
        tenant_id=tenant_id,
        subject=subject,
        predicate=predicate,
        limit=limit,
    )
    
    # Calculate date range
    date_range = {"earliest": None, "latest": None}
    if facts:
        dates = [f.valid_from for f in facts]
        date_range["earliest"] = min(dates).isoformat()
        date_range["latest"] = max(dates).isoformat()
    
    return TimelineResponse(
        subject=subject,
        facts=[fact_to_response(f) for f in facts],
        total_facts=len(facts),
        date_range=date_range,
    )


@router.get("/compare", response_model=CompareResponse)
async def compare_dates(
    subject: str = Query(..., description="Entity to compare"),
    date1: datetime = Query(..., description="First date"),
    date2: datetime = Query(..., description="Second date"),
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """
    Compare what was true about an entity at two different dates.
    
    Returns:
    - added: Facts that exist at date2 but not date1
    - removed: Facts that exist at date1 but not date2
    - changed: Facts with different values between dates
    - unchanged: Facts with same values at both dates
    
    Useful for tracking how knowledge evolves over time.
    """
    tenant_id = user.get("tenant_id", "default")
    
    service = TemporalGraphService(session)
    
    result = await service.compare_dates(
        tenant_id=tenant_id,
        subject=subject,
        date1=date1,
        date2=date2,
    )
    
    return CompareResponse(**result)


@router.get("/volatile", response_model=VolatileFactResponse)
async def get_volatile_facts(
    min_changes: int = Query(3, ge=1, description="Minimum changes to be volatile"),
    time_window_days: int = Query(90, ge=1, le=365, description="Time window in days"),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """
    Find facts that change frequently (volatile facts).
    
    Volatile facts are subject-predicate pairs that have changed
    multiple times within the specified time window.
    
    Useful for identifying:
    - Unstable information
    - Rapidly evolving entities
    - Potentially unreliable data
    """
    tenant_id = user.get("tenant_id", "default")
    
    service = TemporalGraphService(session)
    
    volatile = await service.get_volatile_facts(
        tenant_id=tenant_id,
        min_changes=min_changes,
        time_window_days=time_window_days,
        limit=limit,
    )
    
    return VolatileFactResponse(
        volatile_facts=volatile,
        time_window_days=time_window_days,
        min_changes=min_changes,
    )


@router.get("/stats", response_model=TemporalStatsResponse)
async def get_stats(
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """
    Get statistics about the temporal knowledge graph.
    
    Returns counts of facts, subjects, predicates, and changes.
    """
    tenant_id = user.get("tenant_id", "default")
    
    service = TemporalGraphService(session)
    
    stats = await service.get_stats(tenant_id)
    
    return TemporalStatsResponse(**stats)


@router.post("/decay", status_code=status.HTTP_200_OK)
async def run_decay(
    batch_size: int = Query(1000, ge=100, le=10000),
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """
    Manually trigger confidence decay for temporal facts.
    
    Reduces confidence based on age and predicate type.
    Normally this runs automatically as a background task.
    """
    tenant_id = user.get("tenant_id", "default")
    
    service = TemporalGraphService(session)
    
    updated = await service.decay_confidence(
        tenant_id=tenant_id,
        batch_size=batch_size,
    )
    
    await session.commit()
    
    return {
        "status": "success",
        "facts_updated": updated,
        "message": f"Applied confidence decay to {updated} facts",
    }


@router.get("/memory/{memory_id}/facts", response_model=FactListResponse)
async def get_facts_by_memory(
    memory_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """
    Get all temporal facts extracted from a specific memory.
    
    Useful for tracing back where facts originated from.
    """
    try:
        memory_uuid = UUID(memory_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory_id format"
        )
    
    service = TemporalGraphService(session)
    
    facts = await service.get_facts_by_memory(memory_uuid)
    
    return FactListResponse(
        facts=[fact_to_response(f) for f in facts],
        total=len(facts),
    )



