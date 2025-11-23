"""Memory management endpoints."""

import uuid
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.db.session import get_async_session
from api.dependencies import get_current_user
from core.models.memory import Memory
from core.constants import MemoryType
from services.preprocessor import preprocess_text
from services.antiviral import scan_content
from services.sector_classifier import classify_sector
from api.config import settings
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/v1/memories", tags=["memories"])


class MemoryCreateRequest(BaseModel):
    """Request model for creating a memory.
    
    Note: Memory type is automatically detected by the system.
    Users do not need to specify the type.
    """

    content: str
    tags: list[str] = []
    metadata: Dict[str, Any] = {}


class MemoryResponse(BaseModel):
    """Response model for memory operations."""

    memory_id: str
    status: str
    message: str = "Memory ingested successfully"


@router.post("", response_model=MemoryResponse)
async def create_memory(
    request: MemoryCreateRequest,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """Ingest a new memory with preprocessing.
    
    This endpoint:
    1. Preprocesses and normalizes the text
    2. Redacts PII
    3. Chunks the content
    4. Stores the memory in the database
    
    The memory type (episodic, semantic, emotional, procedural, workspace) is
    automatically detected by the Sector Classifier after embeddings are generated.
    Users do not need to specify the type.
    
    Args:
        request: Memory creation request (content, tags, metadata)
        session: Database session
        user: Current user from JWT token (should contain user_id and tenant_id)
    
    Returns:
        Memory response with memory ID and status
    """
    try:
        # Extract user and tenant information
        # Assuming user dict has 'user_id' and 'tenant_id' keys
        # Adjust based on your JWT token structure
        user_id = user.get("user_id", user.get("sub", "unknown"))
        tenant_id = user.get("tenant_id", user.get("org_id", "default"))
        
        # Step 1: Preprocess the text
        logger.info("Preprocessing memory content", user_id=user_id, content_length=len(request.content))
        
        preprocessed = await preprocess_text(
            text=request.content,
            chunk_size=1000,  # Configurable via settings
            chunk_overlap=200,
            enable_pii_redaction=True
        )
        
        # Step 2: Run ANTIVIRAL scan (if enabled)
        antiviral_result = None
        if settings.enable_antiviral:
            logger.info("Running ANTIVIRAL scan", user_id=user_id)
            antiviral_result = await scan_content(
                content=preprocessed["normalized_text"],
                provenance=None  # Will be set after memory creation
            )
            
            # Handle quarantine action
            if antiviral_result["action"] == "quarantine":
                logger.warning(
                    "Memory quarantined by ANTIVIRAL",
                    memory_id="pending",
                    threat_score=antiviral_result["threat_score"],
                    flags=antiviral_result["flags"][:5]
                )
                # TODO: Store in quarantine store and alert admin
                # For now, we'll still create the memory but mark it as quarantined
        
        # Step 3: Prepare content for storage
        # Store the preprocessed data in the content JSONB field
        content_data = {
            "original_length": len(request.content),
            "normalized_text": preprocessed["normalized_text"],
            "language": preprocessed["language"],
            "chunks": preprocessed["chunks"],
            "pii_redacted": preprocessed["pii_redacted"],
            "pii_counts": preprocessed["pii_counts"],
            "metadata": preprocessed["metadata"],
            "user_metadata": request.metadata,  # User-provided metadata
        }
        
        # Step 4: Classify sector (initial classification, can be refined later with embeddings)
        logger.info("Classifying memory sector", user_id=user_id)
        sector_result = await classify_sector(
            content=preprocessed["normalized_text"],
            embedding=None,  # Will be refined later when embeddings are generated
            metadata=preprocessed["metadata"]
        )
        
        # Step 5: Create memory record
        # Type is auto-detected by Sector Classifier
        # Can be refined later when embeddings are available
        
        # Determine status based on ANTIVIRAL result
        if antiviral_result and antiviral_result["action"] == "quarantine":
            memory_status = "quarantined"
            trust_score = 0.0  # Quarantined memories have zero trust
        elif antiviral_result and antiviral_result["action"] == "low_trust":
            memory_status = "ingested"
            trust_score = max(0.1, 0.5 - antiviral_result["threat_score"])  # Reduce trust
        else:
            memory_status = "ingested"
            trust_score = 0.5  # Default, will be updated after further processing
        
        # Build provenance
        provenance_entry = {
            "source": "api",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "preprocessing": {
                "pii_redacted": preprocessed["pii_redacted"],
                "chunk_count": len(preprocessed["chunks"]),
                "language": preprocessed["language"],
            },
            "note": "Memory type will be auto-detected by Sector Classifier",
        }
        
        # Add ANTIVIRAL results to provenance
        if antiviral_result:
            provenance_entry["antiviral"] = {
                "threat_score": antiviral_result["threat_score"],
                "action": antiviral_result["action"],
                "flags": antiviral_result["flags"][:10],  # Limit flags in provenance
                "breakdown": antiviral_result["breakdown"],
            }
        
        memory = Memory(
            tenant_id=tenant_id,
            user_id=user_id,
            type=sector_result["sector"],  # Auto-detected by sector classifier
            content=content_data,
            tags=request.tags or [],
            status=memory_status,
            importance=0.5,  # Default, will be updated by sector classifier
            trust_score=trust_score,
            provenance=[provenance_entry],
        )
        
        # Add sector classification to provenance
        provenance_entry["sector_classification"] = {
            "sector": sector_result["sector"],
            "confidence": sector_result["confidence"],
            "method": sector_result["method"],
            "scores": sector_result.get("scores", {}),
            "note": "Initial classification, may be refined with embeddings",
        }
        
        # Step 6: Save to database
        session.add(memory)
        await session.commit()
        await session.refresh(memory)
        
        logger.info(
            "Memory ingested successfully",
            memory_id=str(memory.memory_id),
            user_id=user_id,
            tenant_id=tenant_id,
            chunk_count=len(preprocessed["chunks"]),
            pii_redacted=preprocessed["pii_redacted"]
        )
        
        # TODO: Enqueue embedding job to Redis queue
        # TODO: Generate embeddings
        # TODO: Refine sector classification with embeddings (optional - can update memory.type if confidence improves)
        # TODO: Build HSG nodes/edges
        # Note: ANTIVIRAL scan and Sector Classifier are now integrated above
        
        return MemoryResponse(
            memory_id=str(memory.memory_id),
            status=memory.status,
            message="Memory ingested and preprocessed successfully",
        )
        
    except Exception as e:
        logger.error(
            "Failed to ingest memory",
            error=str(e),
            user_id=user_id if 'user_id' in locals() else "unknown"
        )
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest memory: {str(e)}",
        )


@router.get("/{memory_id}")
async def get_memory(
    memory_id: str,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """Get a memory by ID.
    
    Args:
        memory_id: Memory UUID
        session: Database session
        user: Current user from JWT token
    
    Returns:
        Memory record with versions and provenance
    """
    try:
        user_id = user.get("user_id", user.get("sub", "unknown"))
        tenant_id = user.get("tenant_id", user.get("org_id", "default"))
        
        # Parse UUID
        try:
            memory_uuid = uuid.UUID(memory_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid memory ID format",
            )
        
        # Query memory
        result = await session.execute(
            select(Memory).where(
                Memory.memory_id == memory_uuid,
                Memory.tenant_id == tenant_id,  # Ensure tenant isolation
            )
        )
        memory = result.scalar_one_or_none()
        
        if not memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory not found",
            )
        
        # Return memory data
        return {
            "memory_id": str(memory.memory_id),
            "tenant_id": memory.tenant_id,
            "user_id": memory.user_id,
            "type": memory.type,
            "content": memory.content,
            "structured": memory.structured,
            "tags": memory.tags,
            "importance": memory.importance,
            "trust_score": memory.trust_score,
            "version": memory.version,
            "status": memory.status,
            "provenance": memory.provenance,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve memory", error=str(e), memory_id=memory_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory: {str(e)}",
        )