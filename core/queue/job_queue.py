"""Redis queue service for background jobs."""

from typing import Dict, Any, Optional
import json
import redis
from api.config import settings
import structlog

logger = structlog.get_logger()


class JobQueue:
    """Redis-based job queue for background processing."""

    def __init__(self):
        """Initialize Redis connection for job queue (lazy connection)."""
        self.redis_client: Optional[redis.Redis] = None
        self.queue_name = "embedding_jobs"
        self._connection_attempted = False

    def _build_redis_url(self) -> str:
        """Build Redis URL from settings (for logging purposes).
        
        Returns:
            Redis URL with masked password for logging
        """
        # Redis Cloud only supports database index 0
        db_index = 0
        if settings.redis_username and settings.redis_password:
            # Cloud Redis with authentication
            return f"redis://{settings.redis_username}:***@{settings.redis_host}:{settings.redis_port}/{db_index}"
        elif settings.redis_password:
            # Redis with password only
            return f"redis://:***@{settings.redis_host}:{settings.redis_port}/{db_index}"
        else:
            # Local Redis without authentication
            return f"redis://{settings.redis_host}:{settings.redis_port}/{db_index}"

    def _ensure_connected(self) -> bool:
        """Ensure Redis client is connected, creating it if necessary.
        
        Returns:
            True if connected successfully, False otherwise
        """
        if self.redis_client is not None:
            try:
                # Test if connection is still alive
                self.redis_client.ping()
                return True
            except Exception as e:
                logger.warning(
                    "Redis connection lost, reconnecting",
                    error=str(e),
                )
                self.redis_client = None

        # Build Redis URL for connection
        # Redis Cloud only supports database index 0
        db_index = 0
        if settings.redis_username and settings.redis_password:
            # Cloud Redis with authentication
            redis_url = f"redis://{settings.redis_username}:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{db_index}"
        elif settings.redis_password:
            # Redis with password only
            redis_url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{db_index}"
        else:
            # Local Redis without authentication
            redis_url = f"redis://{settings.redis_host}:{settings.redis_port}/{db_index}"

        try:
            logger.info(
                "Connecting to Redis",
                host=settings.redis_host,
                port=settings.redis_port,
                has_username=bool(settings.redis_username),
                has_password=bool(settings.redis_password),
                url_masked=self._build_redis_url(),
            )
            
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test connection immediately
            self.redis_client.ping()
            
            self._connection_attempted = True
            logger.info("Redis connection established successfully")
            return True
            
        except redis.ConnectionError as e:
            self._connection_attempted = True
            error_msg = str(e)
            if "Connection refused" in error_msg or "ECONNREFUSED" in error_msg:
                logger.error(
                    "Redis connection failed: Connection refused. Is Redis running?",
                    host=settings.redis_host,
                    port=settings.redis_port,
                    url_masked=self._build_redis_url(),
                )
            elif "Authentication" in error_msg or "AUTH" in error_msg:
                logger.error(
                    "Redis connection failed: Authentication failed. Check REDIS_USERNAME and REDIS_PASSWORD.",
                    host=settings.redis_host,
                    port=settings.redis_port,
                    has_username=bool(settings.redis_username),
                    has_password=bool(settings.redis_password),
                )
            else:
                logger.error(
                    "Redis connection failed",
                    error=error_msg,
                    host=settings.redis_host,
                    port=settings.redis_port,
                    url_masked=self._build_redis_url(),
                )
            return False
            
        except Exception as e:
            self._connection_attempted = True
            logger.error(
                "Redis connection failed: Unexpected error",
                error=str(e),
                error_type=type(e).__name__,
                host=settings.redis_host,
                port=settings.redis_port,
                url_masked=self._build_redis_url(),
            )
            return False

    def enqueue_embedding_job(
        self,
        memory_id: str,
        tenant_id: str,
        user_id: str,
        content: Dict[str, Any],
        memory_type: str,
        chunks: list,
    ) -> bool:
        """Enqueue an embedding job for a memory.

        Args:
            memory_id: UUID of the memory
            tenant_id: Tenant ID
            user_id: User ID
            content: Memory content dictionary
            memory_type: Memory type (episodic, semantic, etc.)
            chunks: List of text chunks to embed

        Returns:
            True if job was enqueued successfully
        """
        if not self._ensure_connected():
            return False
            
        try:
            job_data = {
                "memory_id": memory_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "content": content,
                "memory_type": memory_type,
                "chunks": chunks,
                "job_type": "embedding",
            }
            
            # Push job to Redis list (FIFO queue)
            self.redis_client.lpush(self.queue_name, json.dumps(job_data))
            
            logger.info(
                "Embedding job enqueued",
                memory_id=memory_id,
                tenant_id=tenant_id,
                memory_type=memory_type,
                chunk_count=len(chunks),
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to enqueue embedding job",
                memory_id=memory_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Mark connection as invalid so it will be recreated next time
            self.redis_client = None
            return False

    def dequeue_embedding_job(self, timeout: int = 0) -> Optional[Dict[str, Any]]:
        """Dequeue an embedding job from the queue.

        Args:
            timeout: Blocking timeout in seconds (0 = non-blocking)

        Returns:
            Job data dictionary or None if no job available
        """
        if not self._ensure_connected():
            return None
            
        try:
            if timeout > 0:
                # Blocking pop with timeout
                result = self.redis_client.brpop(self.queue_name, timeout=timeout)
                if result:
                    _, job_json = result
                    return json.loads(job_json)
            else:
                # Non-blocking pop
                job_json = self.redis_client.rpop(self.queue_name)
                if job_json:
                    return json.loads(job_json)
            return None
        except Exception as e:
            logger.error(
                "Failed to dequeue embedding job",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Mark connection as invalid so it will be recreated next time
            self.redis_client = None
            return None

    def get_queue_length(self) -> int:
        """Get the current queue length.

        Returns:
            Number of jobs in the queue
        """
        if not self._ensure_connected():
            return 0
            
        try:
            return self.redis_client.llen(self.queue_name)
        except Exception as e:
            logger.error(
                "Failed to get queue length",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Mark connection as invalid so it will be recreated next time
            self.redis_client = None
            return 0

    def health_check(self) -> bool:
        """Check if Redis queue is healthy.

        Returns:
            True if Redis is accessible
        """
        if not self._ensure_connected():
            if self._connection_attempted:
                logger.error(
                    "Redis health check failed: Connection not available",
                    host=settings.redis_host,
                    port=settings.redis_port,
                    url_masked=self._build_redis_url(),
                )
            return False
        
        try:
            self.redis_client.ping()
            return True
        except redis.ConnectionError as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "ECONNREFUSED" in error_msg:
                logger.error(
                    "Redis health check failed: Connection refused. Is Redis running?",
                    host=settings.redis_host,
                    port=settings.redis_port,
                )
            elif "Authentication" in error_msg or "AUTH" in error_msg:
                logger.error(
                    "Redis health check failed: Authentication failed. Check REDIS_USERNAME and REDIS_PASSWORD.",
                    host=settings.redis_host,
                    port=settings.redis_port,
                )
            else:
                logger.error(
                    "Redis health check failed: Connection error",
                    error=error_msg,
                    host=settings.redis_host,
                    port=settings.redis_port,
                )
            self.redis_client = None
            return False
        except Exception as e:
            logger.error(
                "Redis health check failed: Unexpected error",
                error=str(e),
                error_type=type(e).__name__,
                host=settings.redis_host,
                port=settings.redis_port,
            )
            self.redis_client = None
            return False

