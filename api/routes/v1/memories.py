"""Memory management endpoints.

Implements the full memory ingestion pipeline:
1. Preprocess and normalize text
2. PII redaction
3. ANTIVIRAL security scan
4. Entity and triple extraction
5. Sector classification
6. HSG node creation with entity linking
7. Embedding job enqueueing
"""

import uuid
from typing import Dict, Any, List
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
from services.extractor import extract_entities_and_triples
from services.hsg_builder import build_hsg_node
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
        
        # Merge user-provided tags with auto-extracted tags
        user_tags = request.tags or []
        extracted_tags = preprocessed.get("extracted_tags", [])
        
        # Combine and deduplicate tags (preserve user tags, add extracted tags)
        all_tags = list(user_tags)  # Start with user tags
        for tag in extracted_tags:
            # Normalize tag for comparison (lowercase)
            tag_lower = tag.lower().strip()
            # Add if not already present (case-insensitive)
            if tag_lower not in [t.lower().strip() for t in all_tags]:
                all_tags.append(tag)
        
        # Remove empty tags and normalize
        final_tags = [tag.strip() for tag in all_tags if tag.strip()]
        
        logger.info(
            "Tags merged",
            user_tags=user_tags,
            extracted_tags=extracted_tags,
            final_tags=final_tags,
            memory_id="pending"
        )
        
        memory = Memory(
            tenant_id=tenant_id,
            user_id=user_id,
            type=sector_result["sector"],  # Auto-detected by sector classifier
            content=content_data,
            tags=final_tags,
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
        
        # Step 7: Enqueue embedding job to Redis queue
        try:
            from core.queue.job_queue import JobQueue
            
            queue = JobQueue()
            if queue.health_check():
                # Extract chunk texts from chunk dictionaries for the queue
                # Chunks are stored as dicts with "text" field, but we need strings for embedding
                chunk_texts = []
                for chunk in preprocessed["chunks"]:
                    if isinstance(chunk, dict):
                        chunk_text = chunk.get("text", "")
                        if chunk_text:
                            chunk_texts.append(chunk_text)
                    elif isinstance(chunk, str):
                        chunk_texts.append(chunk)
                
                queue.enqueue_embedding_job(
                    memory_id=str(memory.memory_id),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    content=content_data,
                    memory_type=sector_result["sector"],
                    chunks=chunk_texts,  # Pass as list of strings
                )
                logger.info(
                    "Embedding job enqueued",
                    memory_id=str(memory.memory_id),
                    sector=sector_result["sector"],
                )
            else:
                logger.warning(
                    "Redis queue not available, embedding job not enqueued",
                    memory_id=str(memory.memory_id),
                )
        except Exception as e:
            logger.error(
                "Failed to enqueue embedding job",
                memory_id=str(memory.memory_id),
                error=str(e),
            )
            # Don't fail the request if queue is unavailable
        
        # Step 8: Extract entities and triples for HSG
        extraction_result = None
        try:
            extraction_result = await extract_entities_and_triples(
                text=preprocessed["normalized_text"],
                calculate_importance_score=True,
            )
            
            logger.info(
                "Entities and triples extracted",
                memory_id=str(memory.memory_id),
                entity_count=len(extraction_result.get("entities", [])),
                triple_count=len(extraction_result.get("triples", [])),
                calculated_importance=extraction_result.get("importance", 0.5),
            )
            
            # Update importance if extraction suggests higher value
            extracted_importance = extraction_result.get("importance", 0.5)
            if extracted_importance > memory.importance:
                memory.importance = extracted_importance
                await session.commit()
                
        except Exception as e:
            logger.warning(
                "Entity extraction failed, continuing without HSG",
                memory_id=str(memory.memory_id),
                error=str(e),
            )
        
        # Step 9: Build HSG node with entities
        hsg_result = None
        if extraction_result:
            try:
                hsg_result = await build_hsg_node(
                    memory_id=str(memory.memory_id),
                    content=content_data,
                    sector=sector_result["sector"],
                    tenant_id=tenant_id,
                    user_id=user_id,
                    importance=memory.importance,
                    trust_score=trust_score,
                    provenance=[provenance_entry],
                    extract_entities=False,  # Already extracted above
                    link_related=True,
                )
                
                # Manually link entities since we already extracted them
                if hsg_result.get("success"):
                    from services.hsg_builder import get_hsg_builder
                    builder = get_hsg_builder()
                    entities_linked = await builder.extract_and_link_entities(
                        memory_id=str(memory.memory_id),
                        entities=extraction_result.get("entities", []),
                    )
                    facts_linked = await builder.link_facts(
                        memory_id=str(memory.memory_id),
                        triples=extraction_result.get("triples", []),
                    )
                    
                    logger.info(
                        "HSG node built successfully",
                        memory_id=str(memory.memory_id),
                        entities_linked=entities_linked,
                        facts_linked=facts_linked,
                        related_linked=hsg_result.get("related_linked", 0),
                    )
                    
            except Exception as e:
                logger.warning(
                    "HSG node creation failed, memory still saved",
                    memory_id=str(memory.memory_id),
                    error=str(e),
                )
        
        return MemoryResponse(
            memory_id=str(memory.memory_id),
            status=memory.status,
            message="Memory ingested, preprocessed, and graph-indexed successfully",
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