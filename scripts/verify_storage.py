"""Verification script for Redis queue and PostgreSQL storage."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select, text
from core.db.session import get_async_session
from core.models.memory import Memory
from core.queue.job_queue import JobQueue
from api.config import settings


async def verify_postgresql():
    """Verify PostgreSQL storage."""
    print("\n" + "=" * 70)
    print("PostgreSQL Storage Verification")
    print("=" * 70)
    
    result = False
    try:
        async for session in get_async_session():
            try:
                # Check if memories table exists
                query_result = await session.execute(
                    text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'memories')")
                )
                table_exists = query_result.scalar()
                
                if not table_exists:
                    print("   [ERROR] 'memories' table does not exist!")
                    result = False
                    break
                
                print("   [OK] 'memories' table exists")
                
                # Count total memories
                query_result = await session.execute(text("SELECT COUNT(*) FROM memories"))
                total_count = query_result.scalar()
                print(f"   [OK] Total memories: {total_count}")
                
                if total_count == 0:
                    print("   [INFO] No memories found in database")
                    result = True
                    break
                
                # Get sample memories with tags
                query_result = await session.execute(
                    select(Memory)
                    .order_by(Memory.created_at.desc())
                    .limit(5)
                )
                memories = query_result.scalars().all()
                
                print(f"\n   Sample memories (showing tags):")
                for i, memory in enumerate(memories, 1):
                    print(f"\n   Memory #{i}:")
                    print(f"     - ID: {memory.memory_id}")
                    print(f"     - Type: {memory.type}")
                    print(f"     - Status: {memory.status}")
                    print(f"     - Tags: {memory.tags if memory.tags else '[]'}")
                    
                    # Show content preview
                    if memory.content and isinstance(memory.content, dict):
                        text_content = memory.content.get('normalized_text', '')
                        if text_content:
                            preview = text_content[:60] + "..." if len(text_content) > 60 else text_content
                            print(f"     - Content: {preview}")
                
                # Check tags statistics
                try:
                    query_result = await session.execute(
                        text("""
                            SELECT 
                                COUNT(*) as total,
                                COUNT(CASE WHEN tags IS NOT NULL AND jsonb_array_length(tags::jsonb) > 0 THEN 1 END) as with_tags,
                                COUNT(CASE WHEN tags IS NULL OR jsonb_array_length(tags::jsonb) = 0 THEN 1 END) as without_tags
                            FROM memories
                        """)
                    )
                    stats = query_result.fetchone()
                    if stats:
                        total, with_tags, without_tags = stats
                        print(f"\n   Tag Statistics:")
                        print(f"     - Total memories: {total}")
                        print(f"     - With tags: {with_tags}")
                        print(f"     - Without tags: {without_tags}")
                except Exception as e:
                    # If JSONB function doesn't work, use simpler query
                    print(f"\n   [INFO] Could not get detailed tag statistics: {e}")
                
                result = True
                break
                
            except Exception as e:
                print(f"   [ERROR] PostgreSQL verification failed: {e}")
                import traceback
                traceback.print_exc()
                result = False
                break
    except Exception as e:
        print(f"   [ERROR] Failed to get database session: {e}")
        result = False
    
    return result


def verify_redis_queue():
    """Verify Redis queue."""
    print("\n" + "=" * 70)
    print("Redis Queue Verification")
    print("=" * 70)
    
    try:
        queue = JobQueue()
        
        # Health check
        if queue.health_check():
            print("   [OK] Redis connection is healthy")
        else:
            print("   [ERROR] Redis connection failed")
            return False
        
        # Queue length
        queue_length = queue.get_queue_length()
        print(f"   [OK] Queue length: {queue_length} jobs")
        
        if queue_length > 0:
            print(f"   [INFO] There are {queue_length} jobs waiting to be processed")
        else:
            print("   [INFO] Queue is empty (no pending jobs)")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Redis verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all verification checks."""
    print("=" * 70)
    print("Storage Verification Script")
    print("=" * 70)
    print(f"\nDatabase: {settings.database_url.split('@')[-1] if '@' in settings.database_url else settings.database_url}")
    print(f"Redis: {settings.redis_host}:{settings.redis_port}")
    
    # Verify PostgreSQL
    pg_ok = await verify_postgresql()
    
    # Verify Redis
    redis_ok = verify_redis_queue()
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    print(f"   PostgreSQL: {'[OK]' if pg_ok else '[FAILED]'}")
    print(f"   Redis Queue: {'[OK]' if redis_ok else '[FAILED]'}")
    
    if pg_ok and redis_ok:
        print("\n   [SUCCESS] All storage systems are working correctly!")
    else:
        print("\n   [WARNING] Some storage systems have issues. Please check the errors above.")
    
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())

