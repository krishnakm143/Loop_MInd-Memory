"""Authentication endpoints for JWT token generation."""

from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from jose import jwt

from api.config import settings

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class TokenRequest(BaseModel):
    """Request model for token generation."""
    
    user_id: str = "test_user"
    tenant_id: str = "default_tenant"


class TokenResponse(BaseModel):
    """Response model for token generation."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    tenant_id: str


@router.post("/token", response_model=TokenResponse)
async def generate_token(request: TokenRequest):
    """Generate a JWT access token for API testing.
    
    This endpoint is for development/testing purposes only.
    In production, tokens should be generated through a proper authentication flow.
    
    Args:
        request: Token request with user_id and tenant_id
        
    Returns:
        Token response with access token and metadata
    """
    # Calculate expiration time
    now = datetime.now(timezone.utc)
    expire = now + timedelta(
        minutes=settings.jwt_access_token_expire_minutes
    )
    
    # Create JWT payload
    payload = {
        "sub": request.user_id,  # Subject (user ID)
        "tenant_id": request.tenant_id,
        "exp": int(expire.timestamp()),  # Expiration time (Unix timestamp)
        "iat": int(now.timestamp()),  # Issued at (Unix timestamp)
    }
    
    # Generate token using python-jose (same library used in authentication)
    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_minutes * 60,  # Convert to seconds
        user_id=request.user_id,
        tenant_id=request.tenant_id,
    )


@router.get("/token/quick")
async def generate_quick_token():
    """Generate a quick test token with default values.
    
    Convenience endpoint for quick token generation during development.
    Uses default user_id="test_user" and tenant_id="default_tenant".
    
    Returns:
        Token response with access token
    """
    request = TokenRequest()
    return await generate_token(request)

