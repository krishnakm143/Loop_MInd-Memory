"""Embedding worker for processing memory embeddings and storing in Qdrant."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.queue.job_queue import JobQueue
from core.db.session import get_async_session
from core.models.memory import Memory
from core.storage.vector import VectorStore
from services.embedding_router import get_embeddings_batch
from api.config import settings
import structlog

logger = structlog.get_logger()


async def process_embedding_job(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process an embedding job.

    This function:
    1. Retrieves the memory from PostgreSQL
    2. Generates embeddings for all chunks
    3. Creates/ensures Qdrant collection exists for the sector
    4. Stores embeddings in Qdrant with metadata
    5. Updates memory status in PostgreSQL

    Args:
        job_data: Job data with memory_id, chunks, etc.

    Returns:
        Processing result with status and details
    """
    memory_id = job_data.get("memory_id")
    tenant_id = job_data.get("tenant_id")
    user_id = job_data.get("user_id")
    memory_type = job_data.get("memory_type", "episodic")
    chunks = job_data.get("chunks", [])
    
    if not memory_id:
        logger.error("Embedding job missing memory_id", job_data=job_data)
        return {"status": "failed", "error": "Missing memory_id"}

    try:
        # Step 1: Get memory from database
        async for session in get_async_session():
            try:
                result = await session.execute(
                    select(Memory).where(Memory.memory_id == uuid.UUID(memory_id))
                )
                memory = result.scalar_one_or_none()
                
                if not memory:
                    logger.error("Memory not found", memory_id=memory_id)
                    return {"status": "failed", "error": "Memory not found"}
                
                # Step 2: Generate embeddings for all chunks
                logger.info(
                    "Generating embeddings",
                    memory_id=memory_id,
                    chunk_count=len(chunks),
                    memory_type=memory_type,
                )
                
                if not chunks:
                    # Fallback: use normalized text if no chunks
                    content = memory.content
                    if isinstance(content, dict):
                        normalized_text = content.get("normalized_text", "")
                        if normalized_text:
                            chunks = [normalized_text]
                
                if not chunks:
                    logger.warning("No chunks to embed", memory_id=memory_id)
                    return {"status": "skipped", "reason": "No chunks to embed"}
                
                # Extract text from chunks if they're dictionaries
                # Chunks can be either strings or dicts with "text" field
                chunk_texts = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        chunk_text = chunk.get("text", "")
                        if chunk_text and chunk_text.strip():
                            chunk_texts.append(chunk_text.strip())
                    elif isinstance(chunk, str) and chunk.strip():
                        chunk_texts.append(chunk.strip())
                
                if not chunk_texts:
                    logger.warning("No valid chunk text to embed", memory_id=memory_id)
                    return {"status": "skipped", "reason": "No valid chunk text to embed"}
                
                # Generate embeddings
                embeddings = await get_embeddings_batch(chunk_texts)
                
                if not embeddings or len(embeddings) != len(chunk_texts):
                    logger.error(
                        "Embedding generation failed",
                        memory_id=memory_id,
                        expected=len(chunk_texts),
                        got=len(embeddings) if embeddings else 0,
                    )
                    return {"status": "failed", "error": "Embedding generation failed"}
                
                # Step 3: Ensure Qdrant collection exists for this sector
                vector_store = VectorStore()
                collection_name = f"memories_{memory_type}"
                
                # Create collection if it doesn't exist
                await vector_store.create_collection(
                    collection_name=collection_name,
                    vector_size=settings.embedding_dimension,
                )
                
                # Step 4: Store embeddings in Qdrant
                from qdrant_client.models import PointStruct
                
                points = []
                for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
                    point_id = f"{memory_id}_{i}"
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "memory_id": str(memory_id),
                            "tenant_id": tenant_id,
                            "user_id": user_id,
                            "chunk_index": i,
                            "chunk_text": chunk_text[:500],  # Store first 500 chars for reference
                            "memory_type": memory_type,
                            "trust_score": float(memory.trust_score),
                            "importance": float(memory.importance),
                            "created_at": memory.created_at.isoformat() if memory.created_at else None,
                        },
                    )
                    points.append(point)
                
                success = await vector_store.upsert_points(
                    collection_name=collection_name,
                    points=points,
                )
                
                if not success:
                    logger.error("Failed to upsert points to Qdrant", memory_id=memory_id)
                    return {"status": "failed", "error": "Failed to store in Qdrant"}
                
                # Step 5: Update memory status
                memory.status = "processed"  # Mark as processed
                
                # Add embedding info to provenance
                if memory.provenance and isinstance(memory.provenance, list):
                    embedding_provenance = {
                        "step": "embedding",
                        "vector_count": len(embeddings),
                        "collection": collection_name,
                        "dimension": settings.embedding_dimension,
                    }
                    memory.provenance.append(embedding_provenance)
                
                await session.commit()
                await session.refresh(memory)
                
                logger.info(
                    "Embedding job completed",
                    memory_id=memory_id,
                    vector_count=len(embeddings),
                    collection=collection_name,
                )
                
                return {
                    "status": "completed",
                    "memory_id": memory_id,
                    "vector_count": len(embeddings),
                    "collection": collection_name,
                }
                
            finally:
                await session.close()
                break
                
    except Exception as e:
        logger.error(
            "Embedding job failed",
            memory_id=memory_id,
            error=str(e),
        )
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


async def run_worker(worker_id: str = "worker-1", poll_interval: int = 1):
    """Run the embedding worker in a loop.

    Args:
        worker_id: Unique identifier for this worker
        poll_interval: Seconds to wait between polling for jobs
    """
    queue = JobQueue()
    
    logger.info("Starting embedding worker", worker_id=worker_id)
    
    # Health check with detailed diagnostics
    logger.info(
        "Checking Redis connection",
        worker_id=worker_id,
        host=settings.redis_host,
        port=settings.redis_port,
        has_username=bool(settings.redis_username),
        has_password=bool(settings.redis_password),
    )
    
    if not queue.health_check():
        logger.error(
            "Redis queue not available - worker cannot start",
            worker_id=worker_id,
            host=settings.redis_host,
            port=settings.redis_port,
            suggestion="Please check: 1) Redis is running, 2) REDIS_HOST and REDIS_PORT are correct, 3) REDIS_USERNAME and REDIS_PASSWORD are set if using cloud Redis",
        )
        return
    
    logger.info(
        "Redis connection verified - worker ready",
        worker_id=worker_id,
    )
    
    while True:
        try:
            # Dequeue a job (blocking with 1 second timeout)
            job_data = queue.dequeue_embedding_job(timeout=1)
            
            if job_data:
                logger.info(
                    "Processing embedding job",
                    worker_id=worker_id,
                    memory_id=job_data.get("memory_id"),
                )
                
                try:
                    result = await process_embedding_job(job_data)
                    
                    # Ensure result is a dict
                    if result is None:
                        logger.error(
                            "process_embedding_job returned None",
                            worker_id=worker_id,
                            memory_id=job_data.get("memory_id"),
                        )
                        result = {"status": "failed", "error": "Job processing returned None"}
                    
                    if result.get("status") == "completed":
                        logger.info(
                            "Job completed successfully",
                            worker_id=worker_id,
                            memory_id=result.get("memory_id"),
                        )
                    else:
                        logger.warning(
                            "Job failed or skipped",
                            worker_id=worker_id,
                            result=result,
                        )
                except Exception as e:
                    logger.error(
                        "Exception while processing embedding job",
                        worker_id=worker_id,
                        memory_id=job_data.get("memory_id"),
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    import traceback
                    traceback.print_exc()
            else:
                # No jobs available, wait a bit
                await asyncio.sleep(poll_interval)
                
        except KeyboardInterrupt:
            logger.info("Worker stopping", worker_id=worker_id)
            break
        except Exception as e:
            logger.error(
                "Worker error",
                worker_id=worker_id,
                error=str(e),
            )
            await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run embedding worker")
    parser.add_argument(
        "--worker-id",
        default=f"worker-{uuid.uuid4().hex[:8]}",
        help="Worker identifier",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=1,
        help="Polling interval in seconds",
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_worker(worker_id=args.worker_id, poll_interval=args.poll_interval))
