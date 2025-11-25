"""Script to check if memories are stored in PostgreSQL database."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select, text
from core.db.session import get_async_session
from core.models.memory import Memory
from api.config import settings


async def main():
    """Check memories in the database."""
    print("=" * 70)
    print("Checking Memories in PostgreSQL Database")
    print("=" * 70)
    print(f"\nDatabase URL: {settings.database_url.split('@')[-1] if '@' in settings.database_url else settings.database_url}")
    
    async for session in get_async_session():
        try:
            # Check if memories table exists
            print("\n1. Checking if 'memories' table exists...")
            result = await session.execute(
                text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'memories')")
            )
            table_exists = result.scalar()
            
            if not table_exists:
                print("   [ERROR] 'memories' table does not exist!")
                print("   Run migrations: alembic upgrade head")
                return
            
            print("   [OK] 'memories' table exists")
            
            # Count total memories
            print("\n2. Counting total memories...")
            result = await session.execute(
                text("SELECT COUNT(*) FROM memories")
            )
            total_count = result.scalar()
            print(f"   Total memories: {total_count}")
            
            if total_count == 0:
                print("\n   [INFO] No memories found in database.")
                print("   This could mean:")
                print("     - No memories have been created yet")
                print("     - Memories were created but not committed")
                print("     - Database connection issue")
                return
            
            # Get recent memories
            print("\n3. Fetching recent memories...")
            result = await session.execute(
                select(Memory)
                .order_by(Memory.created_at.desc())
                .limit(10)
            )
            memories = result.scalars().all()
            
            print(f"\n   Found {len(memories)} recent memories:\n")
            
            for i, memory in enumerate(memories, 1):
                print(f"   Memory #{i}:")
                print(f"     - ID: {memory.memory_id}")
                print(f"     - Type: {memory.type}")
                print(f"     - Status: {memory.status}")
                print(f"     - Created: {memory.created_at}")
                
                # Show content preview
                if memory.content and isinstance(memory.content, dict):
                    text_content = memory.content.get('normalized_text', '')
                    if text_content:
                        preview = text_content[:80] + "..." if len(text_content) > 80 else text_content
                        print(f"     - Content: {preview}")
                
                # Show tags
                if memory.tags:
                    print(f"     - Tags: {memory.tags}")
                
                # Show provenance info
                if memory.provenance and isinstance(memory.provenance, dict):
                    if "sector_classification" in memory.provenance:
                        sc = memory.provenance["sector_classification"]
                        print(f"     - Sector: {sc.get('sector', 'N/A')} (confidence: {sc.get('confidence', 'N/A')})")
                    if "antiviral" in memory.provenance:
                        av = memory.provenance["antiviral"]
                        print(f"     - ANTIVIRAL: {av.get('action', 'N/A')} (threat: {av.get('threat_score', 'N/A')})")
                
                print()
            
            # Show statistics
            print("\n4. Statistics by type:")
            result = await session.execute(
                text("SELECT type, COUNT(*) as count FROM memories GROUP BY type")
            )
            stats = result.fetchall()
            for type_name, count in stats:
                print(f"     - {type_name}: {count}")
            
            # Show statistics by status
            print("\n5. Statistics by status:")
            result = await session.execute(
                text("SELECT status, COUNT(*) as count FROM memories GROUP BY status")
            )
            stats = result.fetchall()
            for status_name, count in stats:
                print(f"     - {status_name}: {count}")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to check database: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await session.close()
            break
    
    print("\n" + "=" * 70)
    print("Note: Memories are stored in PostgreSQL, not in Redis/Qdrant/Neo4j")
    print("=" * 70)
    print("\nStorage breakdown:")
    print("  - PostgreSQL: Primary storage (memories table) - THIS IS WHERE MEMORIES ARE")
    print("  - Qdrant: Vector embeddings (not yet implemented - will store embeddings)")
    print("  - Neo4j: Graph relationships (not yet implemented - will store node/edge relationships)")
    print("  - Redis: Cache and task queue (not primary storage)")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())

