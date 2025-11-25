"""Script to generate JWT tokens for API testing."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.routes.v1.auth import generate_token, TokenRequest
from api.config import settings


async def main():
    """Generate a JWT token for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate JWT token for API testing")
    parser.add_argument(
        "--user-id",
        default="test_user",
        help="User ID for the token (default: test_user)",
    )
    parser.add_argument(
        "--tenant-id",
        default="default_tenant",
        help="Tenant ID for the token (default: default_tenant)",
    )
    parser.add_argument(
        "--secret",
        default=None,
        help="JWT secret key (default: from settings)",
    )
    
    args = parser.parse_args()
    
    # Override secret if provided
    if args.secret:
        settings.jwt_secret_key = args.secret
    
    request = TokenRequest(user_id=args.user_id, tenant_id=args.tenant_id)
    result = await generate_token(request)
    
    print("\n" + "=" * 70)
    print("JWT Token Generated Successfully")
    print("=" * 70)
    print(f"\nToken: {result.access_token}")
    print(f"\nToken Type: {result.token_type}")
    print(f"Expires In: {result.expires_in} seconds ({result.expires_in // 60} minutes)")
    print(f"User ID: {result.user_id}")
    print(f"Tenant ID: {result.tenant_id}")
    print(f"\nSecret Key: {settings.jwt_secret_key}")
    print(f"Algorithm: {settings.jwt_algorithm}")
    print("\n" + "=" * 70)
    print("\nTo use in Swagger UI:")
    print("1. Click the 'Authorize' button (lock icon) at the top right")
    print("2. Enter the token in the 'Value' field")
    print("3. Click 'Authorize'")
    print("4. Click 'Close'")
    print("\nTo use with curl:")
    print(f'curl -H "Authorization: Bearer {result.access_token}" http://localhost:8000/v1/memories')
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())


