"""Analytics API Endpoints.

Provides analytics and usage statistics for the memory system:
- Memory counts and sector distribution
- Query statistics and latency metrics
- Usage trends over time
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.session import get_async_session
from core.models.memory import Memory
from api.dependencies import get_current_user
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/v1/analytics", tags=["analytics"])


# Response Models
class SectorStats(BaseModel):
    """Statistics for a memory sector."""
    sector: str
    count: int
    avg_importance: float
    avg_trust_score: float
    total_size_bytes: int = 0


class OverviewResponse(BaseModel):
    """Overview analytics response."""
    total_memories: int
    active_memories: int
    quarantined_memories: int
    sectors: List[SectorStats]
    created_today: int
    created_this_week: int
    created_this_month: int
    tenant_id: str
    generated_at: datetime


class UsageStats(BaseModel):
    """Usage statistics."""
    period: str
    query_count: int
    memory_created: int
    memory_merged: int
    memory_deleted: int
    avg_query_latency_ms: float
    p95_query_latency_ms: float
    cache_hit_rate: float


class UsageResponse(BaseModel):
    """Usage statistics response."""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    daily_stats: List[UsageStats]
    totals: UsageStats
    generated_at: datetime


class TrendDataPoint(BaseModel):
    """A single data point in a trend."""
    timestamp: datetime
    value: float
    label: str = ""


class TrendResponse(BaseModel):
    """Trend data response."""
    metric: str
    period: str
    data_points: List[TrendDataPoint]
    trend_direction: str  # up, down, stable
    change_percent: float


@router.get("/overview", response_model=OverviewResponse)
async def get_analytics_overview(
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> OverviewResponse:
    """Get analytics overview for the tenant.
    
    Returns:
        - Total memory counts by status
        - Sector distribution with averages
        - Time-based creation counts
    """
    tenant_id = user.get("tenant_id", "default")
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    try:
        # Total counts by status
        total_query = select(func.count(Memory.memory_id)).where(
            Memory.tenant_id == tenant_id
        )
        total_result = await session.execute(total_query)
        total_memories = total_result.scalar() or 0
        
        active_query = select(func.count(Memory.memory_id)).where(
            Memory.tenant_id == tenant_id,
            Memory.status == "active"
        )
        active_result = await session.execute(active_query)
        active_memories = active_result.scalar() or 0
        
        quarantined_query = select(func.count(Memory.memory_id)).where(
            Memory.tenant_id == tenant_id,
            Memory.status == "quarantined"
        )
        quarantined_result = await session.execute(quarantined_query)
        quarantined_memories = quarantined_result.scalar() or 0
        
        # Sector statistics
        sector_query = select(
            Memory.type,
            func.count(Memory.memory_id).label("count"),
            func.avg(Memory.importance).label("avg_importance"),
            func.avg(Memory.trust_score).label("avg_trust_score"),
        ).where(
            Memory.tenant_id == tenant_id
        ).group_by(Memory.type)
        
        sector_result = await session.execute(sector_query)
        sectors = [
            SectorStats(
                sector=row.type or "unknown",
                count=row.count,
                avg_importance=float(row.avg_importance or 0),
                avg_trust_score=float(row.avg_trust_score or 0),
            )
            for row in sector_result
        ]
        
        # Time-based counts
        today_query = select(func.count(Memory.memory_id)).where(
            Memory.tenant_id == tenant_id,
            Memory.created_at >= today
        )
        today_result = await session.execute(today_query)
        created_today = today_result.scalar() or 0
        
        week_query = select(func.count(Memory.memory_id)).where(
            Memory.tenant_id == tenant_id,
            Memory.created_at >= week_ago
        )
        week_result = await session.execute(week_query)
        created_this_week = week_result.scalar() or 0
        
        month_query = select(func.count(Memory.memory_id)).where(
            Memory.tenant_id == tenant_id,
            Memory.created_at >= month_ago
        )
        month_result = await session.execute(month_query)
        created_this_month = month_result.scalar() or 0
        
        logger.info(
            "Analytics overview generated",
            tenant_id=tenant_id,
            total_memories=total_memories,
        )
        
        return OverviewResponse(
            total_memories=total_memories,
            active_memories=active_memories,
            quarantined_memories=quarantined_memories,
            sectors=sectors,
            created_today=created_today,
            created_this_week=created_this_week,
            created_this_month=created_this_month,
            tenant_id=tenant_id,
            generated_at=now,
        )
        
    except Exception as e:
        logger.error("Failed to generate analytics overview", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")


@router.get("/usage", response_model=UsageResponse)
async def get_usage_statistics(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to analyze"),
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> UsageResponse:
    """Get usage statistics for the tenant.
    
    Args:
        days: Number of days to include in statistics
        
    Returns:
        Daily and total usage statistics
    """
    tenant_id = user.get("tenant_id", "default")
    now = datetime.utcnow()
    period_start = now - timedelta(days=days)
    
    try:
        # Get daily memory creation counts
        daily_stats = []
        
        for day_offset in range(days):
            day_start = (now - timedelta(days=day_offset)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            day_end = day_start + timedelta(days=1)
            
            # Memory creation count for the day
            created_query = select(func.count(Memory.memory_id)).where(
                Memory.tenant_id == tenant_id,
                Memory.created_at >= day_start,
                Memory.created_at < day_end
            )
            created_result = await session.execute(created_query)
            created_count = created_result.scalar() or 0
            
            daily_stats.append(UsageStats(
                period=day_start.strftime("%Y-%m-%d"),
                query_count=0,  # Would need query logging table
                memory_created=created_count,
                memory_merged=0,  # Would need merge tracking
                memory_deleted=0,  # Would need delete tracking
                avg_query_latency_ms=0.0,
                p95_query_latency_ms=0.0,
                cache_hit_rate=0.0,
            ))
        
        # Calculate totals
        total_created = sum(s.memory_created for s in daily_stats)
        
        totals = UsageStats(
            period=f"last_{days}_days",
            query_count=0,
            memory_created=total_created,
            memory_merged=0,
            memory_deleted=0,
            avg_query_latency_ms=0.0,
            p95_query_latency_ms=0.0,
            cache_hit_rate=0.0,
        )
        
        logger.info(
            "Usage statistics generated",
            tenant_id=tenant_id,
            days=days,
            total_created=total_created,
        )
        
        return UsageResponse(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=now,
            daily_stats=daily_stats,
            totals=totals,
            generated_at=now,
        )
        
    except Exception as e:
        logger.error("Failed to generate usage statistics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Usage stats error: {str(e)}")


@router.get("/sectors", response_model=List[SectorStats])
async def get_sector_statistics(
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> List[SectorStats]:
    """Get detailed statistics per memory sector.
    
    Returns:
        List of sector statistics with counts and averages
    """
    tenant_id = user.get("tenant_id", "default")
    
    try:
        query = select(
            Memory.type,
            func.count(Memory.memory_id).label("count"),
            func.avg(Memory.importance).label("avg_importance"),
            func.avg(Memory.trust_score).label("avg_trust_score"),
        ).where(
            Memory.tenant_id == tenant_id
        ).group_by(Memory.type)
        
        result = await session.execute(query)
        
        sectors = [
            SectorStats(
                sector=row.type or "unknown",
                count=row.count,
                avg_importance=float(row.avg_importance or 0),
                avg_trust_score=float(row.avg_trust_score or 0),
            )
            for row in result
        ]
        
        # Ensure all sectors are represented
        all_sectors = {"episodic", "semantic", "emotional", "procedural", "workspace"}
        existing = {s.sector for s in sectors}
        for sector in all_sectors - existing:
            sectors.append(SectorStats(
                sector=sector,
                count=0,
                avg_importance=0.0,
                avg_trust_score=0.0,
            ))
        
        return sorted(sectors, key=lambda x: x.count, reverse=True)
        
    except Exception as e:
        logger.error("Failed to get sector statistics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Sector stats error: {str(e)}")


@router.get("/trends/{metric}", response_model=TrendResponse)
async def get_metric_trend(
    metric: str,
    period: str = Query(default="week", regex="^(day|week|month)$"),
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> TrendResponse:
    """Get trend data for a specific metric.
    
    Args:
        metric: Metric to track (memory_count, importance_avg, trust_avg)
        period: Time period (day, week, month)
        
    Returns:
        Trend data with data points and direction
    """
    tenant_id = user.get("tenant_id", "default")
    
    period_days = {"day": 1, "week": 7, "month": 30}[period]
    now = datetime.utcnow()
    
    try:
        data_points = []
        
        # Generate data points for the period
        for i in range(min(period_days, 30)):
            point_time = now - timedelta(days=i)
            point_start = point_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if metric == "memory_count":
                query = select(func.count(Memory.memory_id)).where(
                    Memory.tenant_id == tenant_id,
                    Memory.created_at < point_start + timedelta(days=1)
                )
                result = await session.execute(query)
                value = float(result.scalar() or 0)
                
            elif metric == "importance_avg":
                query = select(func.avg(Memory.importance)).where(
                    Memory.tenant_id == tenant_id
                )
                result = await session.execute(query)
                value = float(result.scalar() or 0)
                
            elif metric == "trust_avg":
                query = select(func.avg(Memory.trust_score)).where(
                    Memory.tenant_id == tenant_id
                )
                result = await session.execute(query)
                value = float(result.scalar() or 0)
                
            else:
                raise HTTPException(status_code=400, detail=f"Unknown metric: {metric}")
            
            data_points.append(TrendDataPoint(
                timestamp=point_start,
                value=value,
                label=point_start.strftime("%Y-%m-%d"),
            ))
        
        # Calculate trend direction
        if len(data_points) >= 2:
            first_value = data_points[-1].value
            last_value = data_points[0].value
            
            if first_value == 0:
                change_percent = 100.0 if last_value > 0 else 0.0
            else:
                change_percent = ((last_value - first_value) / first_value) * 100
            
            if change_percent > 5:
                trend_direction = "up"
            elif change_percent < -5:
                trend_direction = "down"
            else:
                trend_direction = "stable"
        else:
            change_percent = 0.0
            trend_direction = "stable"
        
        return TrendResponse(
            metric=metric,
            period=period,
            data_points=data_points,
            trend_direction=trend_direction,
            change_percent=round(change_percent, 2),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get metric trend", metric=metric, error=str(e))
        raise HTTPException(status_code=500, detail=f"Trend error: {str(e)}")


@router.get("/top-tags")
async def get_top_tags(
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get the most frequently used tags.
    
    Args:
        limit: Maximum number of tags to return
        
    Returns:
        List of tags with counts
    """
    tenant_id = user.get("tenant_id", "default")
    
    try:
        # Query memories and aggregate tags
        query = select(Memory.tags).where(
            Memory.tenant_id == tenant_id,
            Memory.tags.isnot(None)
        )
        result = await session.execute(query)
        
        # Count tag occurrences
        tag_counts: Dict[str, int] = {}
        for row in result:
            if row.tags:
                for tag in row.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort by count and limit
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return {
            "tenant_id": tenant_id,
            "tags": [{"tag": tag, "count": count} for tag, count in sorted_tags],
            "total_unique_tags": len(tag_counts),
        }
        
    except Exception as e:
        logger.error("Failed to get top tags", error=str(e))
        raise HTTPException(status_code=500, detail=f"Tag stats error: {str(e)}")

