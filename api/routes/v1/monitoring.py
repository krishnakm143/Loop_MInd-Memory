"""Monitoring API Endpoints.

Provides monitoring and observability features:
- Comprehensive health checks for all services
- Prometheus-compatible metrics endpoint
- System status and performance metrics
"""

import time
import psutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.session import get_async_session
from core.storage.cache import CacheStore
from core.storage.vector import VectorStore
from core.storage.graph import GraphStore
from api.config import settings
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/v1/monitoring", tags=["monitoring"])


# Metrics storage (in-memory for simplicity, use Redis in production)
_metrics: Dict[str, Any] = {
    "request_count": 0,
    "request_latency_sum": 0.0,
    "request_latency_count": 0,
    "memory_operations": {"create": 0, "read": 0, "update": 0, "delete": 0},
    "query_operations": 0,
    "errors": {"total": 0, "by_type": {}},
    "started_at": datetime.utcnow().isoformat(),
}


class ServiceHealth(BaseModel):
    """Health status of a single service."""
    service: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[float] = None
    details: Optional[str] = None
    last_check: datetime


class HealthResponse(BaseModel):
    """Complete health check response."""
    status: str  # healthy, degraded, unhealthy
    services: List[ServiceHealth]
    uptime_seconds: float
    version: str
    checked_at: datetime


class MetricValue(BaseModel):
    """A single metric value."""
    name: str
    value: float
    type: str  # counter, gauge, histogram
    labels: Dict[str, str] = {}
    help: str = ""


class MetricsResponse(BaseModel):
    """All metrics response."""
    metrics: List[MetricValue]
    collected_at: datetime


async def check_postgres_health(session: AsyncSession) -> ServiceHealth:
    """Check PostgreSQL health."""
    start = time.time()
    try:
        await session.execute(text("SELECT 1"))
        latency = (time.time() - start) * 1000
        return ServiceHealth(
            service="postgresql",
            status="healthy",
            latency_ms=latency,
            last_check=datetime.utcnow(),
        )
    except Exception as e:
        return ServiceHealth(
            service="postgresql",
            status="unhealthy",
            details=str(e)[:100],
            last_check=datetime.utcnow(),
        )


async def check_redis_health() -> ServiceHealth:
    """Check Redis health."""
    start = time.time()
    try:
        cache = CacheStore()
        # Simple ping
        if hasattr(cache.client, 'ping'):
            cache.client.ping()
        latency = (time.time() - start) * 1000
        return ServiceHealth(
            service="redis",
            status="healthy",
            latency_ms=latency,
            last_check=datetime.utcnow(),
        )
    except Exception as e:
        return ServiceHealth(
            service="redis",
            status="unhealthy",
            details=str(e)[:100],
            last_check=datetime.utcnow(),
        )


async def check_qdrant_health() -> ServiceHealth:
    """Check Qdrant health."""
    start = time.time()
    try:
        vector_store = VectorStore()
        collections = vector_store.client.get_collections()
        latency = (time.time() - start) * 1000
        return ServiceHealth(
            service="qdrant",
            status="healthy",
            latency_ms=latency,
            details=f"{len(collections.collections)} collections",
            last_check=datetime.utcnow(),
        )
    except Exception as e:
        return ServiceHealth(
            service="qdrant",
            status="unhealthy",
            details=str(e)[:100],
            last_check=datetime.utcnow(),
        )


async def check_neo4j_health() -> ServiceHealth:
    """Check Neo4j health."""
    start = time.time()
    try:
        graph_store = GraphStore()
        # Simple query to verify connection
        result = await graph_store.execute_query("RETURN 1 as test")
        latency = (time.time() - start) * 1000
        return ServiceHealth(
            service="neo4j",
            status="healthy",
            latency_ms=latency,
            last_check=datetime.utcnow(),
        )
    except Exception as e:
        return ServiceHealth(
            service="neo4j",
            status="unhealthy",
            details=str(e)[:100],
            last_check=datetime.utcnow(),
        )


@router.get("/health", response_model=HealthResponse)
async def get_full_health(
    session: AsyncSession = Depends(get_async_session),
) -> HealthResponse:
    """Get comprehensive health status of all services.
    
    Returns:
        Health status of all components
    """
    start_time = datetime.fromisoformat(_metrics["started_at"])
    uptime = (datetime.utcnow() - start_time).total_seconds()
    
    # Check all services
    services = []
    
    # PostgreSQL
    services.append(await check_postgres_health(session))
    
    # Redis
    services.append(await check_redis_health())
    
    # Qdrant
    services.append(await check_qdrant_health())
    
    # Neo4j
    services.append(await check_neo4j_health())
    
    # Determine overall status
    unhealthy_count = sum(1 for s in services if s.status == "unhealthy")
    degraded_count = sum(1 for s in services if s.status == "degraded")
    
    if unhealthy_count > 0:
        overall_status = "unhealthy"
    elif degraded_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    logger.info(
        "Health check completed",
        status=overall_status,
        unhealthy=unhealthy_count,
        degraded=degraded_count,
    )
    
    return HealthResponse(
        status=overall_status,
        services=services,
        uptime_seconds=uptime,
        version="1.0.0",
        checked_at=datetime.utcnow(),
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get Prometheus-style metrics.
    
    Returns:
        All tracked metrics
    """
    metrics = []
    
    # Request metrics
    metrics.append(MetricValue(
        name="loopmind_requests_total",
        value=float(_metrics["request_count"]),
        type="counter",
        help="Total number of requests processed",
    ))
    
    # Latency metrics
    avg_latency = 0.0
    if _metrics["request_latency_count"] > 0:
        avg_latency = _metrics["request_latency_sum"] / _metrics["request_latency_count"]
    metrics.append(MetricValue(
        name="loopmind_request_latency_avg_ms",
        value=avg_latency,
        type="gauge",
        help="Average request latency in milliseconds",
    ))
    
    # Memory operations
    for op, count in _metrics["memory_operations"].items():
        metrics.append(MetricValue(
            name="loopmind_memory_operations_total",
            value=float(count),
            type="counter",
            labels={"operation": op},
            help="Total memory operations by type",
        ))
    
    # Query operations
    metrics.append(MetricValue(
        name="loopmind_query_operations_total",
        value=float(_metrics["query_operations"]),
        type="counter",
        help="Total query operations",
    ))
    
    # Errors
    metrics.append(MetricValue(
        name="loopmind_errors_total",
        value=float(_metrics["errors"]["total"]),
        type="counter",
        help="Total errors",
    ))
    
    # System metrics
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        metrics.append(MetricValue(
            name="system_cpu_usage_percent",
            value=cpu_percent,
            type="gauge",
            help="System CPU usage percentage",
        ))
        
        metrics.append(MetricValue(
            name="system_memory_usage_percent",
            value=memory.percent,
            type="gauge",
            help="System memory usage percentage",
        ))
        
        metrics.append(MetricValue(
            name="system_memory_available_bytes",
            value=float(memory.available),
            type="gauge",
            help="Available system memory in bytes",
        ))
    except Exception:
        pass  # psutil might not work in all environments
    
    return MetricsResponse(
        metrics=metrics,
        collected_at=datetime.utcnow(),
    )


@router.get("/metrics/prometheus")
async def get_prometheus_metrics() -> Response:
    """Get metrics in Prometheus text format.
    
    Returns:
        Prometheus-compatible metrics output
    """
    metrics_response = await get_metrics()
    
    lines = []
    for metric in metrics_response.metrics:
        # Add help comment
        if metric.help:
            lines.append(f"# HELP {metric.name} {metric.help}")
        lines.append(f"# TYPE {metric.name} {metric.type}")
        
        # Format labels
        if metric.labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in metric.labels.items())
            lines.append(f"{metric.name}{{{label_str}}} {metric.value}")
        else:
            lines.append(f"{metric.name} {metric.value}")
    
    content = "\n".join(lines) + "\n"
    
    return Response(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/status")
async def get_system_status(
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    """Get overall system status summary.
    
    Returns:
        System status and key metrics
    """
    start_time = datetime.fromisoformat(_metrics["started_at"])
    uptime = (datetime.utcnow() - start_time).total_seconds()
    
    # Get memory count
    try:
        from core.models.memory import Memory
        from sqlalchemy import func, select
        
        count_query = select(func.count(Memory.memory_id))
        result = await session.execute(count_query)
        memory_count = result.scalar() or 0
    except Exception:
        memory_count = -1
    
    # System resources
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_resources = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_available_gb": round(disk.free / (1024**3), 2),
        }
    except Exception:
        system_resources = {}
    
    return {
        "status": "operational",
        "version": "1.0.0",
        "uptime_seconds": uptime,
        "uptime_human": _format_uptime(uptime),
        "memory_count": memory_count,
        "request_count": _metrics["request_count"],
        "error_count": _metrics["errors"]["total"],
        "system_resources": system_resources,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable form."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or not parts:
        parts.append(f"{minutes}m")
    
    return " ".join(parts)


# Utility functions for tracking metrics (call from other parts of the app)
def track_request(latency_ms: float):
    """Track a request."""
    _metrics["request_count"] += 1
    _metrics["request_latency_sum"] += latency_ms
    _metrics["request_latency_count"] += 1


def track_memory_operation(operation: str):
    """Track a memory operation."""
    if operation in _metrics["memory_operations"]:
        _metrics["memory_operations"][operation] += 1


def track_query():
    """Track a query operation."""
    _metrics["query_operations"] += 1


def track_error(error_type: str = "unknown"):
    """Track an error."""
    _metrics["errors"]["total"] += 1
    if error_type not in _metrics["errors"]["by_type"]:
        _metrics["errors"]["by_type"][error_type] = 0
    _metrics["errors"]["by_type"][error_type] += 1

