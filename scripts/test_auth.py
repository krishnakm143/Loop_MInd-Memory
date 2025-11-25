"""Test script to verify JWT token authentication."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from jose import jwt, JWTError
from api.config import settings


def test_token(token: str):
    """Test if a token is valid."""
    print("\n" + "=" * 70)
    print("Testing JWT Token")
    print("=" * 70)
    print(f"\nToken: {token[:50]}...")
    print(f"\nSecret Key: {settings.jwt_secret_key[:50]}...")
    print(f"Algorithm: {settings.jwt_algorithm}")
    
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        print("\n[OK] Token is VALID!")
        print(f"\nPayload:")
        print(f"  - User ID (sub): {payload.get('sub')}")
        print(f"  - Tenant ID: {payload.get('tenant_id')}")
        print(f"  - Expires: {payload.get('exp')}")
        print(f"  - Issued At: {payload.get('iat')}")
        return True
    except JWTError as e:
        print(f"\n[ERROR] Token is INVALID!")
        print(f"Error: {str(e)}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_auth.py <token>")
        print("\nExample:")
        print("  python scripts/test_auth.py eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
        sys.exit(1)
    
    token = sys.argv[1].strip()
    # Remove "Bearer " prefix if present
    if token.startswith("Bearer "):
        token = token[7:].strip()
    
    is_valid = test_token(token)
    print("\n" + "=" * 70)
    
    if not is_valid:
        print("\nTip: Generate a new token using:")
        print("   GET http://localhost:8000/v1/auth/token/quick")
        print("   or")
        print("   POST http://localhost:8000/v1/auth/token")
        print("   Body: {\"user_id\": \"test_user\", \"tenant_id\": \"default_tenant\"}")
    
    print("\n")

