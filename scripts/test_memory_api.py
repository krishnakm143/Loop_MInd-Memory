"""Test script to create and retrieve a memory."""

import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx


async def main():
    """Test memory creation and retrieval."""
    base_url = "http://localhost:8000"
    
    print("=" * 70)
    print("Testing Memory API")
    print("=" * 70)
    
    # Step 1: Get a token
    print("\n1. Getting JWT token...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/v1/auth/token/quick")
            response.raise_for_status()
            token_data = response.json()
            token = token_data["access_token"]
            print(f"   [OK] Token obtained: {token[:50]}...")
            print(f"   User ID: {token_data['user_id']}")
            print(f"   Tenant ID: {token_data['tenant_id']}")
        except Exception as e:
            print(f"   [ERROR] Failed to get token: {e}")
            return
    
    # Step 2: Create a memory
    print("\n2. Creating a memory...")
    memory_data = {
        "content": "My flight to London is at 3 PM tomorrow. I need to arrive at Heathrow Airport by 2 PM.",
        "tags": ["travel", "london", "flight"],
        "metadata": {
            "source": "user_input",
            "priority": "high"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{base_url}/v1/memories",
                json=memory_data,
                headers=headers
            )
            response.raise_for_status()
            created_memory = response.json()
            print(f"   [OK] Memory created successfully!")
            print(f"   Response keys: {list(created_memory.keys())}")
            memory_id = created_memory.get("id") or created_memory.get("memory_id") or created_memory.get("uuid")
            if not memory_id:
                print(f"   [WARNING] No ID found in response. Full response:")
                print(json.dumps(created_memory, indent=2))
                return
            print(f"   Memory ID: {memory_id}")
            print(f"   Type: {created_memory.get('type', 'N/A')}")
            print(f"   Status: {created_memory.get('status', 'N/A')}")
            print(f"   Created at: {created_memory.get('created_at', 'N/A')}")
            
            # Show preprocessing results
            if "provenance" in created_memory:
                prov = created_memory["provenance"]
                if "preprocessing" in prov:
                    prep = prov["preprocessing"]
                    print(f"\n   Preprocessing Results:")
                    print(f"     - Language: {prep.get('language', 'N/A')}")
                    print(f"     - Chunks: {len(prep.get('chunks', []))}")
                    print(f"     - PII Redacted: {prep.get('pii_redacted', False)}")
                
                if "antiviral" in prov:
                    av = prov["antiviral"]
                    print(f"\n   ANTIVIRAL Results:")
                    print(f"     - Action: {av.get('action', 'N/A')}")
                    print(f"     - Threat Score: {av.get('threat_score', 'N/A')}")
                
                if "sector_classification" in prov:
                    sc = prov["sector_classification"]
                    print(f"\n   Sector Classification:")
                    print(f"     - Sector: {sc.get('sector', 'N/A')}")
                    print(f"     - Confidence: {sc.get('confidence', 'N/A')}")
            
            print(f"\n   Full Response:")
            print(json.dumps(created_memory, indent=2))
            
        except httpx.HTTPStatusError as e:
            print(f"   [ERROR] HTTP Error: {e.response.status_code}")
            print(f"   Response: {e.response.text}")
            import traceback
            traceback.print_exc()
            return
        except Exception as e:
            print(f"   [ERROR] Failed to create memory: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Step 3: Retrieve the memory
    print(f"\n3. Retrieving memory {memory_id}...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{base_url}/v1/memories/{memory_id}",
                headers=headers
            )
            response.raise_for_status()
            retrieved_memory = response.json()
            print(f"   [OK] Memory retrieved successfully!")
            print(f"\n   Retrieved Memory Details:")
            print(f"     - Memory ID: {retrieved_memory.get('id', 'N/A')}")
            print(f"     - Type: {retrieved_memory.get('type', 'N/A')}")
            print(f"     - Status: {retrieved_memory.get('status', 'N/A')}")
            print(f"     - Tags: {retrieved_memory.get('tags', [])}")
            
            # Show content (it's stored as JSONB)
            content = retrieved_memory.get('content', {})
            if isinstance(content, dict):
                print(f"     - Content (text): {content.get('text', 'N/A')[:100]}...")
                print(f"     - Content (chunks): {len(content.get('chunks', []))} chunks")
            else:
                print(f"     - Content: {str(content)[:100]}...")
            
            # Show provenance if available
            if "provenance" in retrieved_memory:
                prov = retrieved_memory["provenance"]
                print(f"\n   Provenance:")
                if "preprocessing" in prov:
                    prep = prov["preprocessing"]
                    print(f"     - Language: {prep.get('language', 'N/A')}")
                    print(f"     - Chunks: {len(prep.get('chunks', []))}")
                    print(f"     - PII Redacted: {prep.get('pii_redacted', False)}")
                if "antiviral" in prov:
                    av = prov["antiviral"]
                    print(f"     - ANTIVIRAL Action: {av.get('action', 'N/A')}")
                    print(f"     - Threat Score: {av.get('threat_score', 'N/A')}")
                if "sector_classification" in prov:
                    sc = prov["sector_classification"]
                    print(f"     - Sector: {sc.get('sector', 'N/A')}")
                    print(f"     - Confidence: {sc.get('confidence', 'N/A')}")
            
            print(f"\n   Full Retrieved Response:")
            print(json.dumps(retrieved_memory, indent=2, default=str))
            
        except httpx.HTTPStatusError as e:
            print(f"   [ERROR] HTTP Error: {e.response.status_code}")
            print(f"   Response: {e.response.text}")
        except Exception as e:
            print(f"   [ERROR] Failed to retrieve memory: {e}")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

