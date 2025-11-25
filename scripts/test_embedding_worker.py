"""Test script for embedding worker end-to-end."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx
from core.queue.job_queue import JobQueue
from core.storage.vector import VectorStore


async def main():
    """Test embedding worker flow."""
    base_url = "http://localhost:8000"
    
    print("=" * 70)
    print("Testing Embedding Worker - Phase 2")
    print("=" * 70)
    
    # Step 1: Get a token
    print("\n1. Getting JWT token...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/v1/auth/token/quick")
            response.raise_for_status()
            token_data = response.json()
            token = token_data["access_token"]
            print(f"   [OK] Token obtained")
        except Exception as e:
            print(f"   [ERROR] Failed to get token: {e}")
            return
    
    # Step 2: Create a memory
    print("\n2. Creating a memory...")
    memory_data = {
        "content": "I learned Python programming in 2023. It's a versatile language for data science and web development.",
        "tags": ["learning", "python", "programming"],
        "metadata": {"source": "test", "topic": "education"}
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    memory_id = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{base_url}/v1/memories",
                json=memory_data,
                headers=headers
            )
            response.raise_for_status()
            created = response.json()
            memory_id = created["memory_id"]
            print(f"   [OK] Memory created: {memory_id}")
            print(f"   Status: {created.get('status')}")
        except Exception as e:
            print(f"   [ERROR] Failed to create memory: {e}")
            return
    
    # Step 3: Check queue
    print("\n3. Checking Redis queue...")
    queue = JobQueue()
    if queue.health_check():
        queue_length = queue.get_queue_length()
        print(f"   [OK] Queue is healthy")
        print(f"   Queue length: {queue_length}")
    else:
        print(f"   [ERROR] Queue is not healthy")
        return
    
    # Step 4: Process jobs manually (simulate worker)
    print("\n4. Processing embedding job...")
    from workers.embedding_worker import process_embedding_job
    
    # Dequeue and process
    job_data = queue.dequeue_embedding_job(timeout=5)
    if job_data:
        print(f"   [OK] Job dequeued: {job_data.get('memory_id')}")
        result = await process_embedding_job(job_data)
        print(f"   Result: {result.get('status')}")
        if result.get("status") == "completed":
            print(f"   Vector count: {result.get('vector_count')}")
            print(f"   Collection: {result.get('collection')}")
        else:
            print(f"   Error: {result.get('error')}")
    else:
        print(f"   [WARNING] No job found in queue")
        print(f"   This might mean the job was already processed or queue is empty")
    
    # Step 5: Verify in Qdrant
    print("\n5. Verifying embeddings in Qdrant...")
    vector_store = VectorStore()
    if await vector_store.health_check():
        print(f"   [OK] Qdrant is healthy")
        
        # Check collections
        try:
            collections = vector_store.client.get_collections()
            print(f"   Collections found: {len(collections.collections)}")
            for col in collections.collections:
                print(f"     - {col.name}")
                
                # Try to search for our memory
                if memory_id:
                    # We need to know the collection name - it's based on memory type
                    # For now, check all collections
                    for collection_name in [col.name for col in collections.collections]:
                        if "memories_" in collection_name:
                            # Search for points with our memory_id
                            try:
                                from qdrant_client.models import Filter, FieldCondition, MatchValue
                                
                                results = vector_store.client.scroll(
                                    collection_name=collection_name,
                                    scroll_filter=Filter(
                                        must=[
                                            FieldCondition(
                                                key="memory_id",
                                                match=MatchValue(value=str(memory_id))
                                            )
                                        ]
                                    ),
                                    limit=10
                                )
                                
                                points, _ = results
                                if points:  # Points found
                                    print(f"   [OK] Found {len(points)} vectors for memory {memory_id} in {collection_name}")
                                    break
                            except Exception as e:
                                # Collection might not have the filter structure we expect
                                pass
        except Exception as e:
            print(f"   [ERROR] Failed to check collections: {e}")
    else:
        print(f"   [ERROR] Qdrant is not healthy")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Start the worker: python workers/embedding_worker.py")
    print("2. Create more memories to see them processed automatically")
    print("3. Check Qdrant dashboard to see collections and vectors")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())

