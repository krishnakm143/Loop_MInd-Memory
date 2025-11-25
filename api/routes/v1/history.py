"""Memory History/Versioning API Endpoints.

Provides version history and rollback capabilities:
- GET /v1/memories/{id}/history - Get all versions
- GET /v1/memories/{id}/version/{v} - Get specific version
- POST /v1/memories/{id}/revert/{v} - Rollback to version
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.db.session import get_async_session
from core.models.memory import Memory
from api.dependencies import get_current_user
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/v1/memories", tags=["history"])


# In-memory version store (replace with database table in production)
_version_store: Dict[str, List[Dict[str, Any]]] = {}


class MemoryVersion(BaseModel):
    """A single version of a memory."""
    version: int
    content: Any
    tags: List[str]
    type: str
    importance: float
    trust_score: float
    status: str
    created_at: datetime
    modified_by: str
    change_reason: Optional[str] = None


class HistoryResponse(BaseModel):
    """Response for memory history."""
    memory_id: str
    current_version: int
    versions: List[MemoryVersion]
    total_versions: int


class RevertResponse(BaseModel):
    """Response for memory revert."""
    memory_id: str
    reverted_from: int
    reverted_to: int
    new_version: int
    message: str


def save_version(memory: Memory, modified_by: str, change_reason: str = None):
    """Save a version snapshot of a memory."""
    memory_id = str(memory.memory_id)
    
    if memory_id not in _version_store:
        _version_store[memory_id] = []
    
    version_data = {
        "version": memory.version,
        "content": memory.content,
        "tags": memory.tags,
        "type": memory.type,
        "importance": memory.importance,
        "trust_score": memory.trust_score,
        "status": memory.status,
        "created_at": datetime.utcnow().isoformat(),
        "modified_by": modified_by,
        "change_reason": change_reason,
    }
    
    _version_store[memory_id].append(version_data)
    
    logger.info(
        "Memory version saved",
        memory_id=memory_id,
        version=memory.version,
        modified_by=modified_by,
    )


def get_version_history(memory_id: str) -> List[Dict[str, Any]]:
    """Get version history for a memory."""
    return _version_store.get(memory_id, [])


@router.get("/{memory_id}/history", response_model=HistoryResponse)
async def get_memory_history(
    memory_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> HistoryResponse:
    """Get version history for a memory.
    
    Args:
        memory_id: Memory UUID
        
    Returns:
        List of all versions
    """
    tenant_id = user.get("tenant_id", "default")
    
    # Verify memory exists and belongs to tenant
    stmt = select(Memory).where(
        Memory.memory_id == memory_id,
        Memory.tenant_id == tenant_id,
    )
    result = await session.execute(stmt)
    memory = result.scalar_one_or_none()
    
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )
    
    # Get version history
    versions_data = get_version_history(memory_id)
    
    # If no version history, create initial version from current state
    if not versions_data:
        versions_data = [{
            "version": memory.version,
            "content": memory.content,
            "tags": memory.tags,
            "type": memory.type,
            "importance": memory.importance,
            "trust_score": memory.trust_score,
            "status": memory.status,
            "created_at": memory.created_at.isoformat() if memory.created_at else datetime.utcnow().isoformat(),
            "modified_by": memory.user_id,
            "change_reason": "Initial creation",
        }]
    
    versions = [
        MemoryVersion(
            version=v["version"],
            content=v["content"],
            tags=v["tags"],
            type=v["type"],
            importance=v["importance"],
            trust_score=v["trust_score"],
            status=v["status"],
            created_at=datetime.fromisoformat(v["created_at"]) if isinstance(v["created_at"], str) else v["created_at"],
            modified_by=v["modified_by"],
            change_reason=v.get("change_reason"),
        )
        for v in versions_data
    ]
    
    return HistoryResponse(
        memory_id=memory_id,
        current_version=memory.version,
        versions=versions,
        total_versions=len(versions),
    )


@router.get("/{memory_id}/version/{version}", response_model=MemoryVersion)
async def get_memory_version(
    memory_id: str,
    version: int,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> MemoryVersion:
    """Get a specific version of a memory.
    
    Args:
        memory_id: Memory UUID
        version: Version number
        
    Returns:
        Memory state at that version
    """
    tenant_id = user.get("tenant_id", "default")
    
    # Verify memory exists and belongs to tenant
    stmt = select(Memory).where(
        Memory.memory_id == memory_id,
        Memory.tenant_id == tenant_id,
    )
    result = await session.execute(stmt)
    memory = result.scalar_one_or_none()
    
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )
    
    # Get version history
    versions_data = get_version_history(memory_id)
    
    # Find specific version
    version_data = next(
        (v for v in versions_data if v["version"] == version),
        None
    )
    
    if not version_data:
        # If requesting current version, return current state
        if version == memory.version:
            return MemoryVersion(
                version=memory.version,
                content=memory.content,
                tags=memory.tags,
                type=memory.type,
                importance=memory.importance,
                trust_score=memory.trust_score,
                status=memory.status,
                created_at=memory.created_at or datetime.utcnow(),
                modified_by=memory.user_id,
            )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version} not found",
        )
    
    return MemoryVersion(
        version=version_data["version"],
        content=version_data["content"],
        tags=version_data["tags"],
        type=version_data["type"],
        importance=version_data["importance"],
        trust_score=version_data["trust_score"],
        status=version_data["status"],
        created_at=datetime.fromisoformat(version_data["created_at"]) if isinstance(version_data["created_at"], str) else version_data["created_at"],
        modified_by=version_data["modified_by"],
        change_reason=version_data.get("change_reason"),
    )


@router.post("/{memory_id}/revert/{version}", response_model=RevertResponse)
async def revert_memory_version(
    memory_id: str,
    version: int,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> RevertResponse:
    """Revert a memory to a previous version.
    
    Args:
        memory_id: Memory UUID
        version: Version number to revert to
        
    Returns:
        Revert result with new version number
    """
    tenant_id = user.get("tenant_id", "default")
    user_id = user.get("user_id", "unknown")
    
    # Verify memory exists and belongs to tenant
    stmt = select(Memory).where(
        Memory.memory_id == memory_id,
        Memory.tenant_id == tenant_id,
    )
    result = await session.execute(stmt)
    memory = result.scalar_one_or_none()
    
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )
    
    current_version = memory.version
    
    # Get version history
    versions_data = get_version_history(memory_id)
    
    # Find specific version
    version_data = next(
        (v for v in versions_data if v["version"] == version),
        None
    )
    
    if not version_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version} not found",
        )
    
    # Save current state before revert
    save_version(memory, user_id, f"Before revert to version {version}")
    
    # Apply revert
    memory.content = version_data["content"]
    memory.tags = version_data["tags"]
    memory.type = version_data["type"]
    memory.importance = version_data["importance"]
    memory.trust_score = version_data["trust_score"]
    memory.version = current_version + 1
    memory.updated_at = datetime.utcnow()
    
    # Add revert to provenance
    provenance = memory.provenance or []
    provenance.append({
        "action": "revert",
        "from_version": current_version,
        "to_version": version,
        "reverted_by": user_id,
        "timestamp": datetime.utcnow().isoformat(),
    })
    memory.provenance = provenance
    
    await session.commit()
    
    # Save new version
    save_version(memory, user_id, f"Reverted from version {current_version} to {version}")
    
    logger.info(
        "Memory reverted",
        memory_id=memory_id,
        from_version=current_version,
        to_version=version,
        new_version=memory.version,
        user_id=user_id,
    )
    
    return RevertResponse(
        memory_id=memory_id,
        reverted_from=current_version,
        reverted_to=version,
        new_version=memory.version,
        message=f"Successfully reverted from version {current_version} to version {version}",
    )


@router.get("/{memory_id}/diff/{v1}/{v2}")
async def get_version_diff(
    memory_id: str,
    v1: int,
    v2: int,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get diff between two versions of a memory.
    
    Args:
        memory_id: Memory UUID
        v1: First version number
        v2: Second version number
        
    Returns:
        Diff between the two versions
    """
    tenant_id = user.get("tenant_id", "default")
    
    # Verify memory exists and belongs to tenant
    stmt = select(Memory).where(
        Memory.memory_id == memory_id,
        Memory.tenant_id == tenant_id,
    )
    result = await session.execute(stmt)
    memory = result.scalar_one_or_none()
    
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )
    
    # Get version history
    versions_data = get_version_history(memory_id)
    
    # Find both versions
    version1 = next((v for v in versions_data if v["version"] == v1), None)
    version2 = next((v for v in versions_data if v["version"] == v2), None)
    
    if not version1 or not version2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {v1} or {v2} not found",
        )
    
    # Calculate diff
    diff = {
        "memory_id": memory_id,
        "version_1": v1,
        "version_2": v2,
        "changes": [],
    }
    
    # Compare fields
    fields_to_compare = ["content", "tags", "type", "importance", "trust_score", "status"]
    
    for field in fields_to_compare:
        val1 = version1.get(field)
        val2 = version2.get(field)
        
        if val1 != val2:
            diff["changes"].append({
                "field": field,
                "from": val1,
                "to": val2,
            })
    
    return diff

