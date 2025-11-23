"""FastAPI dependencies for authentication and database sessions."""

from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
import os

from core.db.session import get_async_session
from api.config import settings

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session."""
    async for session in get_async_session():
        yield session


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Verify JWT token and return current user (stub implementation).

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        User information dictionary

    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        user_id: str = payload.get("sub")
        tenant_id: str = payload.get("tenant_id")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"user_id": user_id, "tenant_id": tenant_id}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )


async def verify_agent_signature(
    signature: str, message: bytes, public_key_path: Optional[str] = None
) -> bool:
    """Verify Ed25519 signature from agent (stub implementation).

    Args:
        signature: Base64-encoded signature
        message: Message bytes that were signed
        public_key_path: Optional path to public key (defaults to settings)

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        key_path = public_key_path or settings.ed25519_public_key_path
        if not os.path.exists(key_path):
            # In development, allow unsigned requests
            return True

        with open(key_path, "rb") as f:
            public_key_data = f.read()

        public_key = serialization.load_pem_public_key(public_key_data)

        if not isinstance(public_key, ed25519.Ed25519PublicKey):
            return False

        # Decode signature from base64
        import base64

        signature_bytes = base64.b64decode(signature)
        public_key.verify(signature_bytes, message)
        return True
    except Exception:
        return False

