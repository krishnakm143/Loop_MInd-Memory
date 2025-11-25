"""Test creating a memory with auto-tag extraction."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx


async def main():
    """Test memory creation with auto-tags."""
    base_url = "http://localhost:8000"
    
    print("=" * 70)
    print("Testing Memory Creation with Auto-Tag Extraction")
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
    
    # Step 2: Create a memory WITHOUT user tags (to test auto-extraction)
    print("\n2. Creating memory WITHOUT user tags (testing auto-extraction)...")
    memory_data = {
        "content": "I learned Python programming in 2023. It's a versatile language for data science and web development. I also learned JavaScript for frontend work.",
        "tags": [],  # No user tags - should auto-extract
        "metadata": {"source": "test", "topic": "education"}
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
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
    
    # Step 3: Retrieve the memory and check tags
    print(f"\n3. Retrieving memory to check auto-extracted tags...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{base_url}/v1/memories/{memory_id}",
                headers=headers
            )
            response.raise_for_status()
            memory = response.json()
            
            tags = memory.get('tags', [])
            print(f"   [OK] Memory retrieved")
            print(f"   Tags: {tags}")
            
            if tags:
                print(f"   [OK] Auto-tags extracted successfully! Found {len(tags)} tags")
                print(f"   Tag examples: {tags[:5]}")
            else:
                print(f"   [WARNING] No tags found - auto-extraction may not be working")
            
        except Exception as e:
            print(f"   [ERROR] Failed to retrieve memory: {e}")
    
    # Step 4: Create a memory WITH user tags (to test merging)
    print("\n4. Creating memory WITH user tags (testing tag merging)...")
    memory_data2 = {
        "content": "Meeting with John at 2pm to discuss the project deadline and budget allocation.",
        "tags": ["work", "meeting"],  # User-provided tags
        "metadata": {"source": "test"}
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{base_url}/v1/memories",
                json=memory_data2,
                headers=headers
            )
            response.raise_for_status()
            created2 = response.json()
            memory_id2 = created2["memory_id"]
            print(f"   [OK] Memory created: {memory_id2}")
            
            # Retrieve and check merged tags
            response = await client.get(
                f"{base_url}/v1/memories/{memory_id2}",
                headers=headers
            )
            response.raise_for_status()
            memory2 = response.json()
            
            tags2 = memory2.get('tags', [])
            print(f"   User tags: {memory_data2['tags']}")
            print(f"   Final tags (merged): {tags2}")
            
            if len(tags2) > len(memory_data2['tags']):
                print(f"   [OK] Tags merged successfully! User tags + auto-extracted tags")
            elif len(tags2) == len(memory_data2['tags']):
                print(f"   [INFO] Only user tags present (auto-extraction may have found duplicates)")
            else:
                print(f"   [WARNING] Fewer tags than expected")
                
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

