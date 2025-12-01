"""Quick test to debug temporal endpoint."""
import asyncio
import httpx
from datetime import datetime, timezone, timedelta

async def test():
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=30) as client:
        # Get token
        resp = await client.post("/v1/auth/token", json={"user_id": "test", "tenant_id": "test"})
        token = resp.json().get("access_token", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        
        print("Testing temporal endpoint...")
        
        # Try to add a fact
        resp = await client.post(
            "/v1/temporal/facts",
            headers=headers,
            json={
                "subject": "TestCompany",
                "predicate": "has_CEO",
                "object": "Alice",
                "valid_from": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                "confidence": 0.95
            }
        )
        
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        
        if resp.status_code == 500:
            print("\n[ERROR] Server returned 500. Check server logs for details.")

asyncio.run(test())

