"""Rate Limiting Middleware.

Implements per-tenant request quotas and rate limiting:
- Token bucket algorithm for smooth rate limiting
- Per-tenant and per-user limits
- Configurable limits by endpoint
- Redis-backed for distributed deployments
"""

import time
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from fastapi import HTTPException, Request, status, Depends
import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10  # Allow burst of this many requests
    
    # Endpoint-specific limits (multipliers)
    endpoint_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.endpoint_multipliers is None:
            self.endpoint_multipliers = {
                # Heavy endpoints get lower limits
                "/v1/query": 0.5,
                "/v1/memories/batch": 0.3,
                "/v1/export": 0.1,
                # Light endpoints get higher limits
                "/health": 5.0,
                "/v1/monitoring/health": 3.0,
            }


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_at: datetime
    limit: int
    retry_after: Optional[int] = None  # Seconds until reset


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(
        self,
        capacity: int,
        fill_rate: float,  # tokens per second
    ):
        """Initialize token bucket.
        
        Args:
            capacity: Maximum tokens in bucket
            fill_rate: Rate at which tokens are added (per second)
        """
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_update = time.time()
        
    def consume(self, tokens: int = 1) -> Tuple[bool, int]:
        """Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            Tuple of (success, remaining_tokens)
        """
        now = time.time()
        
        # Add tokens based on time elapsed
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        self.last_update = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, int(self.tokens)
        else:
            return False, int(self.tokens)
            
    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens are available."""
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.fill_rate


class InMemoryRateLimiter:
    """In-memory rate limiter for single-instance deployments."""
    
    def __init__(self, default_config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.
        
        Args:
            default_config: Default rate limit configuration
        """
        self.config = default_config or RateLimitConfig()
        self.buckets: Dict[str, TokenBucket] = {}
        self.tenant_configs: Dict[str, RateLimitConfig] = {}
        
    def get_bucket_key(self, tenant_id: str, user_id: str, endpoint: str) -> str:
        """Generate bucket key."""
        return f"{tenant_id}:{user_id}:{endpoint}"
        
    def get_config(self, tenant_id: str) -> RateLimitConfig:
        """Get config for tenant."""
        return self.tenant_configs.get(tenant_id, self.config)
        
    def set_tenant_config(self, tenant_id: str, config: RateLimitConfig):
        """Set custom config for tenant."""
        self.tenant_configs[tenant_id] = config
        logger.info("Rate limit config set", tenant_id=tenant_id)
        
    def get_or_create_bucket(
        self,
        key: str,
        capacity: int,
        fill_rate: float,
    ) -> TokenBucket:
        """Get or create a token bucket."""
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(capacity, fill_rate)
        return self.buckets[key]
        
    def check_rate_limit(
        self,
        tenant_id: str,
        user_id: str,
        endpoint: str,
    ) -> RateLimitResult:
        """Check if request is within rate limits.
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            endpoint: API endpoint being accessed
            
        Returns:
            RateLimitResult indicating if request is allowed
        """
        config = self.get_config(tenant_id)
        
        # Get endpoint multiplier
        multiplier = 1.0
        for pattern, mult in config.endpoint_multipliers.items():
            if pattern in endpoint:
                multiplier = mult
                break
                
        # Calculate effective limits
        effective_rpm = int(config.requests_per_minute * multiplier)
        fill_rate = effective_rpm / 60.0  # tokens per second
        
        # Per-tenant bucket
        tenant_key = f"tenant:{tenant_id}:{endpoint}"
        tenant_bucket = self.get_or_create_bucket(
            tenant_key,
            capacity=effective_rpm * 2,  # Allow 2 minutes worth
            fill_rate=fill_rate,
        )
        
        # Per-user bucket (stricter)
        user_key = f"user:{tenant_id}:{user_id}:{endpoint}"
        user_bucket = self.get_or_create_bucket(
            user_key,
            capacity=config.burst_size,
            fill_rate=fill_rate / 10,  # Users get 1/10th of tenant rate
        )
        
        # Check both buckets
        tenant_ok, tenant_remaining = tenant_bucket.consume()
        user_ok, user_remaining = user_bucket.consume()
        
        allowed = tenant_ok and user_ok
        remaining = min(tenant_remaining, user_remaining)
        
        if not allowed:
            # Calculate retry time
            tenant_wait = tenant_bucket.time_until_available()
            user_wait = user_bucket.time_until_available()
            retry_after = int(max(tenant_wait, user_wait)) + 1
            
            logger.warning(
                "Rate limit exceeded",
                tenant_id=tenant_id,
                user_id=user_id,
                endpoint=endpoint,
                retry_after=retry_after,
            )
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=datetime.utcnow(),
                limit=effective_rpm,
                retry_after=retry_after,
            )
            
        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_at=datetime.utcnow(),
            limit=effective_rpm,
        )
        
    def cleanup_old_buckets(self, max_age_seconds: int = 3600):
        """Clean up buckets that haven't been used recently."""
        now = time.time()
        to_delete = []
        
        for key, bucket in self.buckets.items():
            if now - bucket.last_update > max_age_seconds:
                to_delete.append(key)
                
        for key in to_delete:
            del self.buckets[key]
            
        if to_delete:
            logger.info("Cleaned up rate limit buckets", count=len(to_delete))


class RedisRateLimiter:
    """Redis-backed rate limiter for distributed deployments."""
    
    def __init__(
        self,
        redis_client,
        default_config: Optional[RateLimitConfig] = None,
    ):
        """Initialize Redis rate limiter.
        
        Args:
            redis_client: Redis client instance
            default_config: Default rate limit configuration
        """
        self.redis = redis_client
        self.config = default_config or RateLimitConfig()
        
    def check_rate_limit(
        self,
        tenant_id: str,
        user_id: str,
        endpoint: str,
    ) -> RateLimitResult:
        """Check rate limit using Redis."""
        # Use sliding window counter
        now = int(time.time())
        minute_key = f"ratelimit:{tenant_id}:{user_id}:{endpoint}:{now // 60}"
        
        try:
            # Increment counter
            count = self.redis.incr(minute_key)
            
            # Set expiry on first increment
            if count == 1:
                self.redis.expire(minute_key, 120)  # 2 minute TTL
                
            # Get config
            config = self.config
            multiplier = 1.0
            for pattern, mult in config.endpoint_multipliers.items():
                if pattern in endpoint:
                    multiplier = mult
                    break
                    
            limit = int(config.requests_per_minute * multiplier)
            remaining = max(0, limit - count)
            
            if count > limit:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=datetime.utcnow(),
                    limit=limit,
                    retry_after=60 - (now % 60),
                )
                
            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_at=datetime.utcnow(),
                limit=limit,
            )
            
        except Exception as e:
            # On Redis error, allow request (fail open)
            logger.error("Redis rate limit check failed", error=str(e))
            return RateLimitResult(
                allowed=True,
                remaining=-1,
                reset_at=datetime.utcnow(),
                limit=0,
            )


# Global rate limiter instance
_rate_limiter: Optional[InMemoryRateLimiter] = None


def get_rate_limiter() -> InMemoryRateLimiter:
    """Get the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = InMemoryRateLimiter()
    return _rate_limiter


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""
    
    def __init__(
        self,
        app,
        skip_paths: Optional[list] = None,
        limiter: Optional[InMemoryRateLimiter] = None,
    ):
        """Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            skip_paths: Paths to skip rate limiting
            limiter: Rate limiter instance
        """
        self.app = app
        self.skip_paths = skip_paths or ["/docs", "/openapi.json"]
        self.limiter = limiter or get_rate_limiter()
        
    async def __call__(self, scope, receive, send):
        """Process request through rate limiter."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        path = scope.get("path", "")
        
        # Skip rate limiting for certain paths
        if any(path.startswith(skip) for skip in self.skip_paths):
            await self.app(scope, receive, send)
            return
            
        # Rate limiting happens in the dependency (after auth)
        await self.app(scope, receive, send)


def check_rate_limit(request: Request) -> RateLimitResult:
    """Dependency to check rate limit.
    
    Usage:
        @router.post("/endpoint")
        async def endpoint(
            ...,
            rate_limit: RateLimitResult = Depends(check_rate_limit),
        ):
    """
    # Get user info from request state
    user = getattr(request.state, "user", None)
    
    if user is None:
        # No auth, use IP-based limiting
        tenant_id = "anonymous"
        user_id = request.client.host if request.client else "unknown"
    else:
        tenant_id = user.get("tenant_id", "default")
        user_id = user.get("user_id", "unknown")
        
    limiter = get_rate_limiter()
    result = limiter.check_rate_limit(tenant_id, user_id, request.url.path)
    
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": str(result.remaining),
                "Retry-After": str(result.retry_after or 60),
            },
        )
        
    return result


def rate_limit_headers(result: RateLimitResult) -> Dict[str, str]:
    """Generate rate limit headers for response."""
    return {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        "X-RateLimit-Reset": result.reset_at.isoformat(),
    }

