"""FastAPI dependencies for authentication and database sessions."""

from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
import os
import structlog

from core.db.session import get_async_session
from api.config import settings

logger = structlog.get_logger()

# HTTP Bearer token scheme
# auto_error=False allows us to provide custom error messages
security = HTTPBearer(auto_error=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session."""
    async for session in get_async_session():
        yield session


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Verify JWT token and return current user (stub implementation).

    Args:
        request: FastAPI request object (for manual header parsing fallback)
        credentials: HTTP Bearer token credentials

    Returns:
        User information dictionary

    Raises:
        HTTPException: If token is invalid or expired
    """
    token = None
    
    # Try to get token from HTTPBearer first
    if credentials is not None:
        token = credentials.credentials.strip()
    else:
        # Fallback: manually parse Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()  # Remove "Bearer " prefix
    
    if token is None or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header. Please provide: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        user_id: str = payload.get("sub")
        tenant_id: str = payload.get("tenant_id")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials: missing 'sub' claim",
            )
        logger.debug("JWT token validated", user_id=user_id, tenant_id=tenant_id)
        return {"user_id": user_id, "tenant_id": tenant_id}
    except JWTError as e:
        # Provide more specific error messages
        logger.warning("JWT validation failed", error=str(e), token_preview=token[:20] + "..." if len(token) > 20 else token)
        error_detail = f"Invalid authentication credentials: {str(e)}"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_detail,
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

