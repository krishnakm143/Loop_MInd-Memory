"""Export/Import API Endpoints.

Provides data portability features:
- POST /v1/export - Export memories to JSON/CSV
- POST /v1/import - Import from backup
- GET /v1/export/status/{job_id} - Check export progress
"""

import uuid
import json
import csv
import io
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.db.session import get_async_session
from core.models.memory import Memory
from api.dependencies import get_current_user
from services.preprocessor import preprocess_text
from services.sector_classifier import classify_sector
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/v1", tags=["export"])


# Job storage (use Redis in production)
_export_jobs: Dict[str, Dict[str, Any]] = {}


class ExportRequest(BaseModel):
    """Request for memory export."""
    format: str = Field(default="json", pattern="^(json|csv|jsonl)$")
    filters: Dict[str, Any] = {}
    sectors: List[str] = []
    include_content: bool = True
    include_metadata: bool = True
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class ExportResponse(BaseModel):
    """Response for export request."""
    job_id: str
    status: str
    format: str
    estimated_count: int
    message: str


class ExportStatusResponse(BaseModel):
    """Response for export status check."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    total_records: int
    processed_records: int
    download_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class ImportRequest(BaseModel):
    """Request for memory import."""
    format: str = Field(default="json", pattern="^(json|jsonl)$")
    data: List[Dict[str, Any]]
    overwrite_existing: bool = False
    validate_only: bool = False


class ImportResponse(BaseModel):
    """Response for import request."""
    job_id: str
    status: str
    total: int
    imported: int
    skipped: int
    failed: int
    errors: List[Dict[str, str]]


async def process_export_job(
    job_id: str,
    tenant_id: str,
    request: ExportRequest,
    session: AsyncSession,
):
    """Background task to process export."""
    try:
        _export_jobs[job_id]["status"] = "processing"
        
        # Build query
        stmt = select(Memory).where(Memory.tenant_id == tenant_id)
        
        if request.sectors:
            stmt = stmt.where(Memory.type.in_(request.sectors))
            
        if request.date_from:
            stmt = stmt.where(Memory.created_at >= request.date_from)
            
        if request.date_to:
            stmt = stmt.where(Memory.created_at <= request.date_to)
        
        result = await session.execute(stmt)
        memories = result.scalars().all()
        
        total = len(memories)
        _export_jobs[job_id]["total_records"] = total
        
        # Format data
        export_data = []
        for i, memory in enumerate(memories):
            record = {
                "memory_id": str(memory.memory_id),
                "type": memory.type,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            }
            
            if request.include_content:
                record["content"] = memory.content
                record["tags"] = memory.tags
                
            if request.include_metadata:
                record["importance"] = memory.importance
                record["trust_score"] = memory.trust_score
                record["version"] = memory.version
                record["status"] = memory.status
                record["provenance"] = memory.provenance
            
            export_data.append(record)
            
            # Update progress
            _export_jobs[job_id]["processed_records"] = i + 1
            _export_jobs[job_id]["progress"] = (i + 1) / total if total > 0 else 1.0
        
        # Store export data
        _export_jobs[job_id]["data"] = export_data
        _export_jobs[job_id]["status"] = "completed"
        _export_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        _export_jobs[job_id]["download_url"] = f"/v1/export/download/{job_id}"
        
        logger.info(
            "Export completed",
            job_id=job_id,
            total_records=total,
        )
        
    except Exception as e:
        logger.error("Export failed", job_id=job_id, error=str(e))
        _export_jobs[job_id]["status"] = "failed"
        _export_jobs[job_id]["error"] = str(e)


@router.post("/export", response_model=ExportResponse)
async def create_export(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> ExportResponse:
    """Create an export job.
    
    Args:
        request: Export configuration
        
    Returns:
        Export job ID and status
    """
    job_id = str(uuid.uuid4())
    tenant_id = user.get("tenant_id", "default")
    
    # Estimate count
    stmt = select(Memory).where(Memory.tenant_id == tenant_id)
    if request.sectors:
        stmt = stmt.where(Memory.type.in_(request.sectors))
    result = await session.execute(stmt)
    estimated_count = len(result.scalars().all())
    
    # Create job record
    _export_jobs[job_id] = {
        "status": "pending",
        "format": request.format,
        "progress": 0.0,
        "total_records": estimated_count,
        "processed_records": 0,
        "created_at": datetime.utcnow().isoformat(),
        "tenant_id": tenant_id,
        "data": None,
    }
    
    # Queue background task
    background_tasks.add_task(
        process_export_job,
        job_id,
        tenant_id,
        request,
        session,
    )
    
    logger.info(
        "Export job created",
        job_id=job_id,
        format=request.format,
        estimated_count=estimated_count,
    )
    
    return ExportResponse(
        job_id=job_id,
        status="pending",
        format=request.format,
        estimated_count=estimated_count,
        message="Export job created. Check status at /v1/export/status/{job_id}",
    )


@router.get("/export/status/{job_id}", response_model=ExportStatusResponse)
async def get_export_status(
    job_id: str,
    user: dict = Depends(get_current_user),
) -> ExportStatusResponse:
    """Get export job status.
    
    Args:
        job_id: Export job ID
        
    Returns:
        Current status of the export job
    """
    tenant_id = user.get("tenant_id", "default")
    
    if job_id not in _export_jobs:
        raise HTTPException(
            status_code=404,
            detail="Export job not found",
        )
    
    job = _export_jobs[job_id]
    
    # Verify tenant
    if job.get("tenant_id") != tenant_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied",
        )
    
    return ExportStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        total_records=job["total_records"],
        processed_records=job["processed_records"],
        download_url=job.get("download_url"),
        error=job.get("error"),
        created_at=datetime.fromisoformat(job["created_at"]),
        completed_at=datetime.fromisoformat(job["completed_at"]) if job.get("completed_at") else None,
    )


@router.get("/export/download/{job_id}")
async def download_export(
    job_id: str,
    user: dict = Depends(get_current_user),
):
    """Download exported data.
    
    Args:
        job_id: Export job ID
        
    Returns:
        Exported data as file download
    """
    tenant_id = user.get("tenant_id", "default")
    
    if job_id not in _export_jobs:
        raise HTTPException(
            status_code=404,
            detail="Export job not found",
        )
    
    job = _export_jobs[job_id]
    
    if job.get("tenant_id") != tenant_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied",
        )
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Export not ready. Status: {job['status']}",
        )
    
    data = job["data"]
    format_type = job["format"]
    
    if format_type == "json":
        content = json.dumps(data, indent=2, default=str)
        media_type = "application/json"
        filename = f"export_{job_id}.json"
        
    elif format_type == "jsonl":
        content = "\n".join(json.dumps(record, default=str) for record in data)
        media_type = "application/x-ndjson"
        filename = f"export_{job_id}.jsonl"
        
    elif format_type == "csv":
        if data:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            for record in data:
                # Flatten nested fields
                flat_record = {}
                for k, v in record.items():
                    if isinstance(v, (dict, list)):
                        flat_record[k] = json.dumps(v)
                    else:
                        flat_record[k] = v
                writer.writerow(flat_record)
            content = output.getvalue()
        else:
            content = ""
        media_type = "text/csv"
        filename = f"export_{job_id}.csv"
    
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )


@router.post("/import", response_model=ImportResponse)
async def import_memories(
    request: ImportRequest,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> ImportResponse:
    """Import memories from backup.
    
    Args:
        request: Import data and configuration
        
    Returns:
        Import results
    """
    job_id = str(uuid.uuid4())
    tenant_id = user.get("tenant_id", "default")
    user_id = user.get("user_id", "unknown")
    
    imported = 0
    skipped = 0
    failed = 0
    errors: List[Dict[str, str]] = []
    
    for i, record in enumerate(request.data):
        try:
            # Validate required fields
            if "content" not in record:
                errors.append({
                    "index": str(i),
                    "error": "Missing required field: content",
                })
                failed += 1
                continue
            
            # Check if memory already exists
            original_id = record.get("memory_id")
            if original_id:
                stmt = select(Memory).where(
                    Memory.memory_id == original_id,
                    Memory.tenant_id == tenant_id,
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()
                
                if existing and not request.overwrite_existing:
                    skipped += 1
                    continue
                    
                if existing and request.overwrite_existing:
                    # Update existing
                    existing.content = record["content"]
                    existing.tags = record.get("tags", existing.tags)
                    existing.type = record.get("type", existing.type)
                    existing.version = existing.version + 1
                    existing.updated_at = datetime.utcnow()
                    imported += 1
                    continue
            
            # Validation only mode
            if request.validate_only:
                imported += 1
                continue
            
            # Preprocess content
            content = record["content"]
            if isinstance(content, str):
                preprocessed = await preprocess_text(content)
                content_text = preprocessed["normalized_text"]
            else:
                content_text = json.dumps(content)
            
            # Classify sector if not provided
            sector = record.get("type")
            if not sector:
                sector_result = await classify_sector(content_text)
                sector = sector_result["sector"]
            
            # Create new memory
            memory = Memory(
                tenant_id=tenant_id,
                user_id=user_id,
                type=sector,
                content=record["content"],
                tags=record.get("tags", []),
                status=record.get("status", "active"),
                importance=record.get("importance", 0.5),
                trust_score=record.get("trust_score", 1.0),
                version=1,
                provenance=[{
                    "source": "import",
                    "job_id": job_id,
                    "original_id": original_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "imported_by": user_id,
                }],
            )
            
            session.add(memory)
            imported += 1
            
        except Exception as e:
            logger.error(
                "Import record failed",
                job_id=job_id,
                index=i,
                error=str(e),
            )
            errors.append({
                "index": str(i),
                "error": str(e)[:100],
            })
            failed += 1
    
    if not request.validate_only:
        await session.commit()
    
    logger.info(
        "Import completed",
        job_id=job_id,
        total=len(request.data),
        imported=imported,
        skipped=skipped,
        failed=failed,
        validate_only=request.validate_only,
    )
    
    return ImportResponse(
        job_id=job_id,
        status="completed" if failed == 0 else "completed_with_errors",
        total=len(request.data),
        imported=imported,
        skipped=skipped,
        failed=failed,
        errors=errors[:10],  # Limit error details
    )

